from functools import reduce

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tensorflow.keras import backend

from tfkeras import FreqFusion

from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization, RegressTranslation, \
    CalculateTxTy, GroupNormalization
from initializers import PriorProbability
from utils.anchors import anchors_for_shape
import numpy as np

MOMENTUM = 0.997
EPSILON = 1e-4


def build_Net(phi,
              num_classes=12,
              num_anchors=9,
              freeze_bn=False,
              score_threshold=0.5,
              anchor_parameters=None,
              num_rotation_parameters=6,
              print_architecture=True):
    """
    Builds a model
    Args:
        phi: Scaling hyperparameter phi
        num_classes: Number of classes,
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        anchor_parameters: Struct containing anchor parameters. If None, default values are used.
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation e.g. 3 for r_6d representation
        print_architecture: Boolean indicating if the model architecture should be printed or not

    Returns:
        train: model without NMS used for training
        prediction: model including NMS used for evaluating and inferencing
        all_layers: List of all layers in the model to load weights. Otherwise it can happen that a subnet is considered as a single unit when loading weights and if the output dimension doesn't match with the weight file, the whole subnet weight loading is skipped
    """

    # select parameters according to the given phi
    assert phi in range(8)
    scaled_parameters = get_scaled_parameters(phi)

    input_size = scaled_parameters["input_size"]
    input_shape = (input_size, input_size, 3)
    bifpn_width = subnet_width = scaled_parameters["bifpn_width"]
    bifpn_depth = scaled_parameters["bifpn_depth"]
    subnet_depth = scaled_parameters["subnet_depth"]
    subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
    num_groups_gn = scaled_parameters["num_groups_gn"]
    backbone_class = scaled_parameters["backbone_class"]

    # input layers
    image_input = layers.Input(input_shape)

    camera_parameters_input = layers.Input(
        (6,))  # camera parameters and image scale for calculating the translation vector from 2D x-, y-coordinates

    # build FPN_backbone
    backbone_feature_maps = backbone_class(input_tensor=image_input, freeze_bn=freeze_bn)

    # build FPN
    fpn_feature_maps = build_FPN(backbone_feature_maps, bifpn_width, freeze_bn)

    # build subnets
    box_net, class_net, rotation_net, translation_net = build_subnets(num_classes,
                                                                      subnet_width,
                                                                      subnet_depth,
                                                                      subnet_num_iteration_steps,
                                                                      num_groups_gn,
                                                                      num_rotation_parameters,
                                                                      freeze_bn,
                                                                      num_anchors)

    # apply subnets to feature maps
    classification, bbox_regression, rotation, translation, transformation, bboxes = apply_subnets_to_feature_maps(
        box_net,
        class_net,
        rotation_net,
        translation_net,
        fpn_feature_maps,
        image_input,
        camera_parameters_input,
        input_size,
        anchor_parameters)

    # get the model for training without NMS and the rotation and translation output combined in the transformation output because of the loss calculation
    net_train = models.Model(inputs=[image_input, camera_parameters_input],
                                       outputs=[classification, bbox_regression, transformation], name='netpose')

    # filter detections (apply NMS / score threshold / select top-k)
    filtered_detections = FilterDetections(num_rotation_parameters=num_rotation_parameters,
                                           num_translation_parameters=3,
                                           name='filtered_detections',
                                           score_threshold=score_threshold
                                           )([bboxes, classification, rotation, translation])

    net_prediction = models.Model(inputs=[image_input, camera_parameters_input], outputs=filtered_detections,
                                            name='netpose_prediction')

    if print_architecture:
        print_models(net_train, box_net, class_net, rotation_net, translation_net)
    
    # create list with all layers to be able to load all layer weights because sometimes the whole subnet weight loading is skipped if the output shape does not match instead of skipping just the output layer
    all_layers = list(
        set(net_train.layers + box_net.layers + class_net.layers + rotation_net.layers + translation_net.layers))

    return net_train, net_prediction, all_layers


def get_scaled_parameters(phi):
    """
    Get all needed scaled parameters to build Net
    Args:
        phi: Net scaling hyperparameter phi

    Returns:
       Dictionary containing the scaled parameters
    """
    # info tuples with scalable parameters
    image_sizes = (512, 512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 64, 88, 112, 160, 224, 288, 384)
    bifpn_depths = (3, 3, 4, 5, 6, 7, 7, 8)
    subnet_depths = (3, 3, 3, 3, 4, 4, 4, 5)
    subnet_iteration_steps = (1, 1, 1, 1, 2, 2, 2, 3)
    num_groups_gn = (4, 4, 4, 7, 10, 14, 18, 24)  # try to get 16 channels per group
    backbones = (FreqFusion, FreqFusion, FreqFusion,FreqFusion,FreqFusion, FreqFusion,FreqFusion,FreqFusion)

    parameters = {"input_size": image_sizes[phi],
                  "bifpn_width": bifpn_widths[phi],
                  "bifpn_depth": bifpn_depths[phi],
                  "subnet_depth": subnet_depths[phi],
                  "subnet_num_iteration_steps": subnet_iteration_steps[phi],
                  "num_groups_gn": num_groups_gn[phi],
                  "backbone_class": backbones[phi]}

    return parameters


def build_FPN(backbone_feature_maps, fpn_width, freeze_bn):
    fpn_feature_maps = backbone_feature_maps
    fpn_feature_maps = build_FPN_layer(fpn_feature_maps, fpn_width, freeze_bn=freeze_bn)
    return fpn_feature_maps


def build_FPN_layer(features, num_channels, freeze_bn=False):
    C2, C3, C4, C5 = features

    P2_in, P3_in, P4_in, P5_in = my_prepare_feature_maps_for_FPN(C2, C3, C4, C5, num_channels, freeze_bn)

    # top down pathway
    P5_out, P4_out, P3_out, P2_out = my_top_down_pathway_FPN(P5_in, P4_in, P3_in, P2_in, num_channels)

    # bottom up pathway
    P2_out, P3_out, P4_out, P5_out = my_bottom_up_pathway_FPN(P2_out, P3_out, P4_out, P5_out, num_channels)

    P5_out = layers.Conv2D(num_channels, 1, strides=(1, 1), padding='same', name='p5')(P5_out)
    P4_out = layers.Conv2D(num_channels, 1, strides=(1, 1), padding='same', name='p4')(P4_out)
    P3_out = layers.Conv2D(num_channels, 1, strides=(1, 1), padding='same', name='p3')(P3_out)
    P2_out = layers.Conv2D(num_channels, 1, strides=(1, 1), padding='same', name='p2')(P2_out)
    P6_out = layers.Conv2D(num_channels, 1, strides=(2, 2), padding='same', name='p6')(P5_out)

    return P2_out, P3_out, P4_out, P5_out, P6_out  # TODO check if it is a bug to return the top down feature maps instead of the output maps


def my_prepare_feature_maps_for_FPN(C2, C3, C4, C5, num_channels, freeze_bn):
    P2_in = C2
    P2_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                          name='fpn_cells/cell_0/fnode3/resample_0_0_8/conv2d')(P2_in)
    P2_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                               name='fpn_cells/cell_0/fnode3/resample_0_0_8/bn')(P2_in)
    P3_in = C3
    P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                          name='fpn_cells/cell_0/fnode3/resample_0_1_7/conv2d')(P3_in)
    P3_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                               name='fpn_cells/cell_0/fnode3/resample_0_1_7/bn')(P3_in)

    P4_in = C4
    P4_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                          name='fpn_cells/cell_0/fnode2/resample_0_2_6/conv2d')(P4_in)
    P4_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                               name='fpn_cells/cell_0/fnode2/resample_0_2_6/bn')(P4_in)

    P5_in = C5
    P5_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                          name='fpn_cells/cell_0/fnode1/resample_0_3_5/conv2d')(P5_in)
    P5_in = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                               name='fpn_cells/cell_0/fnode1/resample_0_3_5/bn')(P5_in)

    return P2_in, P3_in, P4_in, P5_in


def my_top_down_pathway_FPN(P5_in, P4_in, P3_in, P2_in, num_channels):
    P5_out = P5_in

    feature_map_resampled = layers.UpSampling2D(size=(2, 2))(P5_out)
    P4_out = layers.Add()([feature_map_resampled, P4_in])
    P4_out = layers.Conv2D(
        filters=num_channels,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        name=f'fpn_cells/cell_1/fnode0/op_after_combine4_conv'
    )(P4_out)
    
    P4_out = BatchNormalization(freeze=False, momentum=MOMENTUM, epsilon=EPSILON,
        name=f'fpn_cells/cell_1/fnode0/op_after_combine4_bn'
    )(P4_out)
    
    P4_out = layers.Activation('relu', name='fpn_cells/cell_1/fnode0/op_after_combine4_relu')(P4_out)
    P4_out = cbam_block(P4_out)

    feature_map_resampled = layers.UpSampling2D(size=(2, 2))(P4_out)
    P3_out = layers.Add()([feature_map_resampled, P3_in])
    P3_out = layers.Conv2D(
        filters=num_channels,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        name=f'fpn_cells/cell_1/fnode0/op_after_combine3_conv'
    )(P3_out)
    
    P3_out = BatchNormalization(freeze=False, momentum=MOMENTUM, epsilon=EPSILON,
        name=f'fpn_cells/cell_1/fnode0/op_after_combine3_bn'
    )(P3_out)
    
    P3_out = layers.Activation('relu', name='fpn_cells/cell_1/fnode0/op_after_combine3_relu')(P3_out)
    P3_out = cbam_block(P3_out)

    feature_map_resampled = layers.UpSampling2D(size=(2, 2))(P3_out)
    P2_out = layers.Add()([feature_map_resampled, P2_in])
    P2_out = layers.Conv2D(
        filters=num_channels,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        name=f'fpn_cells/cell_1/fnode0/op_after_combine2_conv'
    )(P2_out)
    
    P2_out = BatchNormalization(freeze=False, momentum=MOMENTUM, epsilon=EPSILON,
        name=f'fpn_cells/cell_1/fnode0/op_after_combine2_bn'
    )(P2_out)
    
    P2_out = layers.Activation('relu', name='fpn_cells/cell_1/fnode0/op_after_combine2_relu')(P2_out)
    P2_out = cbam_block(P2_out)

    return P5_out, P4_out, P3_out, P2_out


def my_bottom_up_pathway_FPN(P2_in, P3_in, P4_in, P5_in, num_channels):
    P2_out = P2_in

    feature_map_resampled = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P2_out)
    P3_out = layers.Add()([feature_map_resampled, P3_in])
    P3_out = layers.Conv2D(
        filters=num_channels,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        name=f'fpn_cells/cell_1/fnode1/op_after_combine3_conv'
    )(P3_out)
    
    P3_out = BatchNormalization(freeze=False, momentum=MOMENTUM, epsilon=EPSILON,
        name=f'fpn_cells/cell_1/fnode1/op_after_combine3_bn'
    )(P3_out)
    
    P3_out = layers.Activation('relu', name='fpn_cells/cell_1/fnode1/op_after_combine3_relu')(P3_out)
    P3_out = cbam_block(P3_out)

    feature_map_resampled = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
    P4_out = layers.Add()([feature_map_resampled, P4_in])
    P4_out = layers.Conv2D(
        filters=num_channels,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        name=f'fpn_cells/cell_1/fnode1/op_after_combine4_conv'
    )(P4_out)
    
    P4_out = BatchNormalization(freeze=False, momentum=MOMENTUM, epsilon=EPSILON,
        name=f'fpn_cells/cell_1/fnode1/op_after_combine4_bn'
    )(P4_out)
    
    P4_out = layers.Activation('relu', name='fpn_cells/cell_1/fnode1/op_after_combine4_relu')(P4_out)
    P4_out = cbam_block(P4_out)

    feature_map_resampled = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
    P5_out = layers.Add()([feature_map_resampled, P5_in])
    P5_out = layers.Conv2D(
        filters=num_channels,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        name=f'fpn_cells/cell_1/fnode1/op_after_combine5_conv'
    )(P5_out)
    
    P5_out = BatchNormalization(freeze=False, momentum=MOMENTUM, epsilon=EPSILON,
        name=f'fpn_cells/cell_1/fnode1/op_after_combine5_bn'
    )(P5_out)
    
    P5_out = layers.Activation('relu', name='fpn_cells/cell_1/fnode1/op_after_combine5_relu')(P5_out)
    P5_out = cbam_block(P5_out)

    return P2_out, P3_out, P4_out, P5_out


def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = layers.Dense(channel // ratio,
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')
    shared_layer_two = layers.Dense(channel,
                                    kernel_initializer='he_normal',
                                    use_bias=True,
                                    bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(filters=1,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='sigmoid',
                                 kernel_initializer='he_normal',
                                 use_bias=False)(concat)

    return layers.Multiply()([input_feature, cbam_feature])

def cbam_block(input_feature):
    cbam_feature = channel_attention(input_feature)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

    
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters=64,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   activation='sigmoid',
                                   kernel_initializer='he_normal',
                                   use_bias=False)
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.concat = layers.Concatenate(axis=-1)
        self.conv2 = layers.Conv2D(filters=1,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   activation='sigmoid',
                                   kernel_initializer='he_normal',
                                   use_bias=False)

    def call(self, input_feature):
        x = self.conv1(input_feature)
        x = self.bn(x)
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_out = self.concat([avg_out, max_out])
        spatial_out = self.conv2(spatial_out)
        return spatial_out


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    """
    Builds a small block consisting of a depthwise separable convolution layer and a batch norm layer
    Args:
        num_channels: Number of channels used in the BiFPN
        kernel_size: Kernel site of the depthwise separable convolution layer
        strides: Stride of the depthwise separable convolution layer
        name: Name of the block
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.

    Returns:
       The depthwise separable convolution block
    """
    f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same', use_bias=True,
                                name=f'{name}/conv')
    f2 = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def build_subnets(num_classes, subnet_width, subnet_depth, subnet_num_iteration_steps, num_groups_gn,
                  num_rotation_parameters, freeze_bn, num_anchors):
    """
    Builds the subnetworks
    Args:
        num_classes: Number of classes for the classification network output
        subnet_width: The number of channels used in the subnetwork layers
        subnet_depth: The number of layers used in the subnetworks
        subnet_num_iteration_steps: The number of iterative refinement steps used in the rotation and translation subnets
        num_groups_gn: The number of groups per group norm layer used in the rotation and translation subnets
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors

    Returns:
       The subnetworks
    """
    box_net = BoxNet(subnet_width,
                     subnet_depth,
                     num_anchors=num_anchors,
                     freeze_bn=freeze_bn,
                     name='box_net')

    class_net = ClassNet(subnet_width,
                         subnet_depth,
                         num_classes=num_classes,
                         num_anchors=num_anchors,
                         freeze_bn=freeze_bn,
                         name='class_net')

    rotation_net = RotationNet(subnet_width,
                               subnet_depth,
                               num_values=num_rotation_parameters,
                               num_iteration_steps=subnet_num_iteration_steps,
                               num_anchors=num_anchors,
                               freeze_bn=freeze_bn,
                               use_group_norm=True,
                               num_groups_gn=num_groups_gn,
                               name='rotation_net')

    translation_net = TranslationNet(subnet_width,
                                     subnet_depth,
                                     num_iteration_steps=subnet_num_iteration_steps,
                                     num_anchors=num_anchors,
                                     freeze_bn=freeze_bn,
                                     use_group_norm=True,
                                     num_groups_gn=num_groups_gn,
                                     name='translation_net')

    return box_net, class_net, rotation_net, translation_net


class BoxNet(models.Model):
    def __init__(self, width, depth, num_anchors=9, freeze_bn=False, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = 4

        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.convs = [layers.SeparableConv2D(filters=self.width, name=f'{self.name}/box-{i}', **options) for i in
                      range(self.depth)]
        self.head = layers.SeparableConv2D(filters=self.num_anchors * self.num_values, name=f'{self.name}/box-predict',
                                           **options)

        self.bns = [[BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                                        name=f'{self.name}/box-{i}-bn-{j}') for j in range(3, 8)] for i in
                    range(self.depth)]
        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, self.num_values))
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        self.level += 1
        return outputs


class ClassNet(models.Model):
    def __init__(self, width, depth, num_classes=8, num_anchors=9, freeze_bn=False, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [
            layers.SeparableConv2D(filters=self.width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                   **options) for i in range(self.depth)]
        self.head = layers.SeparableConv2D(filters=self.num_classes * self.num_anchors,
                                           bias_initializer=PriorProbability(probability=0.01),
                                           name=f'{self.name}/class-predict', **options)

        self.bns = [[BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                                        name=f'{self.name}/class-{i}-bn-{j}') for j in range(3, 8)] for i in
                    range(self.depth)]
        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, self.num_classes))
        self.activation_sigmoid = layers.Activation('sigmoid')
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation_sigmoid(outputs)
        self.level += 1
        return outputs

class my_IterativeRotationSubNet(models.Model):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors=9, freeze_bn=False,
                 use_group_norm=True, num_groups_gn=None, **kwargs):
        super(my_IterativeRotationSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn

        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1

        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters=width, name=f'{self.name}/iterative-rotation-sub-{i}', **options)
                      for i in range(self.depth)]
        self.pre_conv = layers.SeparableConv2D(filters=width, name=f'{self.name}/iterative-rotation-pre-sub', **options)
        self.head = layers.SeparableConv2D(filters=self.num_anchors * self.num_values,
                                           name=f'{self.name}/iterative-rotation-sub-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis,
                                                    name=f'{self.name}/iterative-rotation-sub-{k}-{i}-gn-{j}') for j in
                                 range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
            self.pre_norm_layer = GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis,
                                                     name=f'{self.name}/iterative-rotation-pre-gn-sub')
        else:
            self.norm_layer = [[[BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                                                    name=f'{self.name}/iterative-rotation-sub-{k}-{i}-bn-{j}') for j in
                                 range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
            self.pre_norm_layer = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                                                     name=f'{self.name}/iterative-rotation-pre-bn-sub')

        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.spaatt = SpatialAttention()
        self.multiply = layers.Multiply()

        self.add = layers.Add()
        self.concat = layers.Concatenate(axis=channel_axis)

    def call(self, inputs, **kwargs):
        feature, rotation, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        # 拼接处理后的 feature 和 rotation
        combined_feature = self.concat([feature, rotation])
        combined_feature = self.pre_conv(combined_feature)
        combined_feature = self.pre_norm_layer(combined_feature)
        combined_feature = self.activation(combined_feature)

        spa = self.spaatt(combined_feature)
        f = self.multiply([feature, spa])
        feature = self.add([feature, f])
        #feature = self.add([feature, feature])

        feature = self.concat([feature, rotation])

        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)

        return outputs

class RotationNet(models.Model):
    def __init__(self, width, depth, num_values, num_iteration_steps, num_anchors=9, freeze_bn=False,
                 use_group_norm=True, num_groups_gn=None, **kwargs):
        super(RotationNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn

        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1

        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters=self.width, name=f'{self.name}/rotation-{i}', **options) for i in
                      range(self.depth)]
        self.initial_rotation = layers.SeparableConv2D(filters=self.num_anchors * self.num_values,
                                                       name=f'{self.name}/rotation-init-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis,
                                                   name=f'{self.name}/rotation-{i}-gn-{j}') for j in range(3, 8)] for i
                               in range(self.depth)]
        else:
            self.norm_layer = [[BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                                                   name=f'{self.name}/rotation-{i}-bn-{j}') for j in range(3, 8)] for i
                               in range(self.depth)]

        self.iterative_submodel = my_IterativeRotationSubNet(width=self.width,
                                                          depth=self.depth - 1,
                                                          num_values=self.num_values,
                                                          num_iteration_steps=self.num_iteration_steps,
                                                          num_anchors=self.num_anchors,
                                                          freeze_bn=freeze_bn,
                                                          use_group_norm=self.use_group_norm,
                                                          num_groups_gn=self.num_groups_gn,
                                                          name="iterative_rotation_subnet")

        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_values))
        self.level = 0
        self.add = layers.Add()
        #self.concat = layers.Concatenate(axis=channel_axis)

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[i][self.level](feature)
            feature = self.activation(feature)

        rotation = self.initial_rotation(feature)

        for i in range(self.num_iteration_steps):
            # iterative_input = self.concat([feature, rotation])
            # delta_rotation = self.iterative_submodel([iterative_input, level], level_py=self.level, iter_step_py=i)
            delta_rotation = self.iterative_submodel([feature, rotation, level], level_py=self.level, iter_step_py=i)
            rotation = self.add([rotation, delta_rotation])

        outputs = self.reshape(rotation)
        self.level += 1
        return outputs

class my_IterativeTranslationSubNet(models.Model):
    def __init__(self, width, depth, num_iteration_steps, num_anchors=9, freeze_bn=False, use_group_norm=True,
                 num_groups_gn=None, **kwargs):
        super(my_IterativeTranslationSubNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn

        if backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1

        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.convs = [
            layers.SeparableConv2D(filters=self.width, name=f'{self.name}/iterative-translation-sub-{i}', **options) for
            i in range(self.depth)]
        self.pre_conv = layers.SeparableConv2D(filters=width, name=f'{self.name}/iterative-translation-pre-sub',
                                               **options)
        self.head_xy = layers.SeparableConv2D(filters=self.num_anchors * 2,
                                              name=f'{self.name}/iterative-translation-xy-sub-predict', **options)
        self.head_z = layers.SeparableConv2D(filters=self.num_anchors,
                                             name=f'{self.name}/iterative-translation-z-sub-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[[GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis,
                                                    name=f'{self.name}/iterative-translation-sub-{k}-{i}-gn-{j}') for j
                                 in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
            self.pre_norm_layer = GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis,
                                                     name=f'{self.name}/iterative-translation-pre-gn-sub')
        else:
            self.norm_layer = [[[BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                                                    name=f'{self.name}/iterative-translation-sub-{k}-{i}-bn-{j}') for j
                                 in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
            self.pre_norm_layer = BatchNormalization(freeze=freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
                                                     name=f'{self.name}/iterative-translation-pre-bn-sub')

        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))

        self.spaatt = SpatialAttention()
        self.multiply = layers.Multiply()
        self.add = layers.Add()
        self.concat = layers.Concatenate(axis=-1 if backend.image_data_format() == 'channels_last' else 1)

    def call(self, inputs, **kwargs):
        feature, translation_xy, translation_z, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]

        # 拼接处理后的 feature 和 rotation
        combined_feature = self.concat([feature, translation_xy, translation_z])
        combined_feature = self.pre_conv(combined_feature)
        combined_feature = self.pre_norm_layer(combined_feature)
        combined_feature = self.activation(combined_feature)

        spa = self.spaatt(combined_feature)
        f = self.multiply([feature, spa])
        feature = self.add([feature, f])

        feature = self.concat([feature, translation_xy, translation_z])

        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        outputs_xy = self.head_xy(feature)
        outputs_z = self.head_z(feature)

        return outputs_xy, outputs_z

class TranslationNet(models.Model):
    def __init__(self, width, depth, num_iteration_steps, num_anchors=9, freeze_bn=False, use_group_norm=True,
                 num_groups_gn=None, **kwargs):
        super(TranslationNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn

        if backend.image_data_format() == 'channels_first':
            channel_axis = 0
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1

        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        self.convs = [layers.SeparableConv2D(filters=self.width, name=f'{self.name}/translation-{i}', **options) for i
                      in range(self.depth)]
        self.initial_translation_xy = layers.SeparableConv2D(filters=self.num_anchors * 2,
                                                             name=f'{self.name}/translation-xy-init-predict', **options)
        self.initial_translation_z = layers.SeparableConv2D(filters=self.num_anchors,
                                                            name=f'{self.name}/translation-z-init-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis,
                                                   name=f'{self.name}/translation-{i}-gn-{j}') for j in range(3, 8)] for
                               i in range(self.depth)]
        else:
            self.norm_layer = [[layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON,
                                                          name=f'{self.name}/translation-{i}-bn-{j}') for j in
                                range(3, 8)] for i in range(self.depth)]

        self.iterative_submodel = my_IterativeTranslationSubNet(width=self.width,
                                                             depth=self.depth - 1,
                                                             num_iteration_steps=self.num_iteration_steps,
                                                             num_anchors=self.num_anchors,
                                                             freeze_bn=freeze_bn,
                                                             use_group_norm=self.use_group_norm,
                                                             num_groups_gn=self.num_groups_gn,
                                                             name="iterative_translation_subnet")

        self.activation = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape_xy = layers.Reshape((-1, 2))
        self.reshape_z = layers.Reshape((-1, 1))
        self.level = 0
        self.add = layers.Add()
        #self.concat = layers.Concatenate(axis=channel_axis)

        self.concat_output = layers.Concatenate(axis=-1)  # always last axis after reshape

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[i][self.level](feature)
            feature = self.activation(feature)

        translation_xy = self.initial_translation_xy(feature)
        translation_z = self.initial_translation_z(feature)

        for i in range(self.num_iteration_steps):
            # iterative_input = self.concat([feature, translation_xy, translation_z])
            # delta_translation_xy, delta_translation_z = self.iterative_submodel(
            #     [iterative_input, level],
            #     level_py=self.level, iter_step_py=i)
            delta_translation_xy, delta_translation_z = self.iterative_submodel([feature, translation_xy, translation_z, level],
                                                                                level_py=self.level, iter_step_py=i)
            translation_xy = self.add([translation_xy, delta_translation_xy])
            translation_z = self.add([translation_z, delta_translation_z])

        outputs_xy = self.reshape_xy(translation_xy)
        outputs_z = self.reshape_z(translation_z)
        outputs = self.concat_output([outputs_xy, outputs_z])
        self.level += 1
        return outputs

def apply_subnets_to_feature_maps(box_net, class_net, rotation_net, translation_net, fpn_feature_maps, image_input,
                                  camera_parameters_input, input_size, anchor_parameters):
    classification = [class_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    classification = layers.Concatenate(axis=1, name='classification')(classification)  # 将结果合并

    bbox_regression = [box_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    bbox_regression = layers.Concatenate(axis=1, name='regression')(bbox_regression)  # 将结果合并

    rotation = [rotation_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    rotation = layers.Concatenate(axis=1, name='rotation')(rotation)

    translation_raw = [translation_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    translation_raw = layers.Concatenate(axis=1, name='translation_raw_outputs')(translation_raw)

    # get anchors and apply predicted translation offsets to translation anchors
    anchors, translation_anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    translation_anchors_input = np.expand_dims(translation_anchors, axis=0)

    translation_xy_Tz = RegressTranslation(name='translation_regression')([translation_anchors_input, translation_raw])
    translation = CalculateTxTy(name='translation')(translation_xy_Tz,
                                                    fx=camera_parameters_input[:, 0],
                                                    fy=camera_parameters_input[:, 1],
                                                    px=camera_parameters_input[:, 2],
                                                    py=camera_parameters_input[:, 3],
                                                    tz_scale=camera_parameters_input[:, 4],
                                                    image_scale=camera_parameters_input[:, 5])

    # apply predicted 2D bbox regression to anchors
    anchors_input = np.expand_dims(anchors, axis=0)
    bboxes = RegressBoxes(name='boxes')([anchors_input, bbox_regression[..., :4]])
    bboxes = ClipBoxes(name='clipped_boxes')([image_input, bboxes])

    # concat rotation and translation outputs to transformation output to have a single output for transformation loss calculation
    # standard concatenate layer throws error that shapes does not match because translation shape dim 2 is known via translation_anchors and rotation shape dim 2 is None
    # so just use lambda layer with tf concat
    transformation = layers.Lambda(lambda input_list: tf.concat(input_list, axis=-1), name="transformation")(
        [rotation, translation])

    return classification, bbox_regression, rotation, translation, transformation, bboxes


def print_models(*models):
    """
    Print the model architectures
    Args:
        *models: Tuple containing all models that should be printed
    """
    for model in models:
        print("\n\n")
        model.summary()
        print("\n\n")
