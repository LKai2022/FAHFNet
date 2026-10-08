from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_core.python.keras.layers.pooling import MaxPooling2D

import os
import json
import math
import string
import collections
import numpy as np

from six.moves import xrange
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import preprocess_input as _preprocess_input

from utils import get_submodules_from_kwargs
from layers import BatchNormalization


import tensorflow as tf

backend = None
layers = None
models = None
keras_utils = None

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def preprocess_input(x, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in ['backend', 'layers', 'models', 'utils']}
    return _preprocess_input(x, mode='torch', **kwargs)


def get_swish(**kwargs):
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    def swish(x):

        if backend.backend() == 'tensorflow':
            try:
                return backend.tf.nn.swish(x)
            except AttributeError:
                pass

        return x * backend.sigmoid(x)

    return swish


def get_dropout(**kwargs):
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.
    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.

    Issue:
        https://github.com/tensorflow/tensorflow/issues/30946
    """
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout


def ConvBlock(input_tensor, num_output, stride, stage_name, block_name):
    filter1, filter2 = num_output

    x = layers.Conv2D(filter1, 3, strides=stride, padding='same', name='res' + stage_name + block_name + '_branch2a')(
        input_tensor)
    x = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2a')(x)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res' + stage_name + block_name + '_branch2b')(x)
    x = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2b')(x)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_branch2b_relu')(x)

    shortcut = layers.Conv2D(filter2, 1, strides=stride, padding='same',
                             name='res' + stage_name + block_name + '_branch1')(input_tensor)
    shortcut = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch1')(shortcut)

    x = layers.add([x, shortcut], name='res' + stage_name + block_name)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_relu')(x)

    return x

def IdentityBlock(input_tensor, num_output, stage_name, block_name):
    filter1, filter2 = num_output

    x = layers.Conv2D(filter1, 3, strides=(1, 1), padding='same', name='res' + stage_name + block_name + '_branch2a')(
        input_tensor)
    x = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2a')(x)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res' + stage_name + block_name + '_branch2b')(x)
    x = layers.BatchNormalization(name='bn' + stage_name + block_name + '_branch2b')(x)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_branch2b_relu')(x)

    shortcut = input_tensor

    x = layers.add([x, shortcut], name='res' + stage_name + block_name)
    x = layers.Activation('relu', name='res' + stage_name + block_name + '_relu')(x)

    return x

class GELU(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(GELU, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return 0.5 * inputs * (1 + tf.tanh(tf.sqrt(2 / tf.constant(3.141592653589793)) * (inputs + 0.044715 * tf.pow(inputs, 3))))

    def get_config(self):
        return super(GELU, self).get_config()  # 确保该层可以被序列化

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, name=None, **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.wq = layers.Dense(embed_dim, name=f"{self.name}_wq")
        self.wk = layers.Dense(embed_dim, name=f"{self.name}_wk")
        self.wv = layers.Dense(embed_dim, name=f"{self.name}_wv")
        self.dense = layers.Dense(embed_dim, name=f"{self.name}_dense")

    def call(self, inputs):
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        scaled_attention = self.scaled_dot_product_attention(q, k, v)
        concat_attention = self.combine_heads(scaled_attention)
        output = self.dense(concat_attention)
        return output

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.embed_dim))

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, name=None, **kwargs):
        super(TransformerBlock, self).__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.att = MultiHeadAttention(embed_dim, num_heads, name=f"{self.name}_att")
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, name=f"{self.name}_mlp_dense1"),
            GELU(name=f"{self.name}_mlp_gelu"),
            layers.Dropout(dropout, name=f"{self.name}_mlp_dropout1"),
            layers.Dense(embed_dim, name=f"{self.name}_mlp_dense2"),
            layers.Dropout(dropout, name=f"{self.name}_mlp_dropout2")
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_layernorm1")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_layernorm2")
        self.dropout1 = layers.Dropout(dropout, name=f"{self.name}_dropout1")
        self.dropout2 = layers.Dropout(dropout, name=f"{self.name}_dropout2")

    def call(self, inputs, **kwargs):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + inputs
        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output)
        return mlp_output + out1

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim, name="position_embedding", **kwargs):
        super(PositionEmbedding, self).__init__(name=name, **kwargs)
        self.embedding = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        pos_emb = self.embedding(positions)
        return inputs + pos_emb

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        return config
    
    # 封装成一个 Keras 层，避免重复创建
class TransformerFeatureProcessor(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, stage_name, dropout=0.1, **kwargs):
        super(TransformerFeatureProcessor, self).__init__(name=f"transformer_processor_{stage_name}", **kwargs)
        self.stage_name = stage_name
        self.position_embedding = None  # 位置编码层将在 build 过程中创建
        self.transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, name=f"transformer_block_{stage_name}")

    def build(self, input_shape):
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        self.position_embedding = PositionEmbedding(sequence_length=height * width, embed_dim=channels, name=f"position_embedding_{self.stage_name}")

    def call(self, x):
        batch_size, height, width, channels = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]

        # 将 4D 特征图转换为 2D 序列
        x_seq = tf.reshape(x, (batch_size, height * width, channels))

        # 添加位置编码
        x_seq = self.position_embedding(x_seq)

        # 通过 Transformer 处理
        x_transformed = self.transformer_block(x_seq)

        # 变回 4D 特征图
        x_output = tf.reshape(x_transformed, (batch_size, height, width, channels))
        return x_output

    def get_config(self):
        config = super(TransformerFeatureProcessor, self).get_config()
        config.update({"stage_name": self.stage_name})
        return config

def attention_mechanism(x1, x2, x3):
    summed_features = x1 + x2 + x3

    gap_output = layers.GlobalAveragePooling2D()(summed_features)
    gap_output = tf.expand_dims(tf.expand_dims(gap_output, 1), 1)

    d = 64
    fc1_output = layers.Dense(d, activation='relu')(gap_output)

    fc2_output = layers.Dense(3 * x1.shape[-1])(fc1_output)
    fc2_output = tf.reshape(fc2_output, [-1, 1, 1, 3, x1.shape[-1]])

    attention_weights = tf.nn.softmax(fc2_output, axis=3)

    x1_weighted = x1 * attention_weights[:, :, :, 0, :]
    x2_weighted = x2 * attention_weights[:, :, :, 1, :]
    x3_weighted = x3 * attention_weights[:, :, :, 2, :]

    final_output = x1_weighted + x2_weighted + x3_weighted
    return final_output

def spatial_attention(x1, x2):
    concat_features = layers.Concatenate(axis=-1)([x1, x2])

    avg_pool = tf.reduce_mean(concat_features, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(concat_features, axis=-1, keepdims=True)
    pool_concat = tf.concat([avg_pool, max_pool], axis=-1)
    spatial_att = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(pool_concat)

    return spatial_att

def channel_attention(x1, x2):
    summed_features = x1 + x2

    gap_output = layers.GlobalAveragePooling2D()(summed_features)
    gap_output = tf.expand_dims(tf.expand_dims(gap_output, 1), 1)

    d = 64
    fc1_output = layers.Dense(d, activation='relu')(gap_output)
    fc2_output = layers.Dense(2 * x1.shape[-1])(fc1_output)
    fc2_output = tf.reshape(fc2_output, [-1, 1, 1, 2, x1.shape[-1]])

    attention_weights = tf.nn.softmax(fc2_output, axis=3)

    x1_weighted = x1 * attention_weights[:, :, :, 0, :]
    x2_weighted = x2 * attention_weights[:, :, :, 1, :]

    final_output = x1_weighted + x2_weighted
    return final_output

def feature_fusion(x1, x2):
    att = spatial_attention(x1, x2)
    new_x1 = x1 + att * x2
    new_x2 = x2 + att * x1
    x = channel_attention(new_x1, new_x2)
    final_x = layers.Conv2D(x1.shape[-1], kernel_size=1, padding='same')(x)
    return final_x

def freqbackboneNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 model_name='fpnnet',
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 freeze_bn=False,
                 **kwargs):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    features = []

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_resolution,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.python.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    activation = get_swish(**kwargs)

    # Build stem
    x = img_input
    x = layers.Conv2D(64, 3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = BatchNormalization(freeze=freeze_bn, axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)
    x = MaxPooling2D((3,3),strides=2,padding='same')(x)

    # Build blocks
    # conv2_x
    x1 = ConvBlock(input_tensor=x, num_output=(64, 64), stride=(2, 2), stage_name='2', block_name='a')
    x2 = IdentityBlock(input_tensor=x1, num_output=(64, 64), stage_name='2', block_name='b')
    x3 = IdentityBlock(input_tensor=x2, num_output=(64, 64), stage_name='2', block_name='c')
    x = attention_mechanism(x1, x2, x3)
    features.append(x)

    # conv3_x
    x1 = ConvBlock(input_tensor=x, num_output=(128, 128), stride=(2, 2), stage_name='3', block_name='a')
    x2 = IdentityBlock(input_tensor=x1, num_output=(128, 128), stage_name='3', block_name='b')
    x3 = IdentityBlock(input_tensor=x2, num_output=(128, 128), stage_name='3', block_name='c')
    x = attention_mechanism(x1, x2, x3)
    features.append(x)

    # conv4_x
    x1 = ConvBlock(input_tensor=x, num_output=(256, 256), stride=(2, 2), stage_name='4', block_name='a')
    x2 = IdentityBlock(input_tensor=x1, num_output=(256, 256), stage_name='4', block_name='b')
    x3 = IdentityBlock(input_tensor=x2, num_output=(256, 256), stage_name='4', block_name='c')
    x = attention_mechanism(x1, x2, x3)
    transformer_processor = TransformerFeatureProcessor(embed_dim=x3.shape[-1], num_heads=4, mlp_dim=256, stage_name="01")
    x4 = transformer_processor(x3)
    x = feature_fusion(x, x4)
    features.append(x)

    # conv5_x
    x1 = ConvBlock(input_tensor=x, num_output=(512, 512), stride=(2, 2), stage_name='5', block_name='a')
    x2 = IdentityBlock(input_tensor=x1, num_output=(512, 512), stage_name='5', block_name='b')
    x3 = IdentityBlock(input_tensor=x2, num_output=(512, 512), stage_name='5', block_name='c')
    x = attention_mechanism(x1, x2, x3)
    transformer_processor = TransformerFeatureProcessor(embed_dim=x3.shape[-1], num_heads=4, mlp_dim=256, stage_name="02")
    x4 = transformer_processor(x3)
    x = feature_fusion(x, x4)
    features.append(x)
    return features

def FreqFusion(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return freqbackboneNet(2.0, 3.1, 600, 0.5,
                        model_name='fpn-backbone-net',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


setattr(FreqFusion, '__doc__', freqbackboneNet.__doc__)
