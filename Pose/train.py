import argparse
import time
import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from model import build_Net
from losses import smooth_l1, focal, transformation_rot_loss

from custom_load_weights import custom_load_weights


def parse_args(args):
    """
    Parse the arguments.
    """
    date_and_time = time.strftime("%d_%m_%Y_%H_%M_%S")
    parser = argparse.ArgumentParser(description = 'Simple training script.')
    subparsers = parser.add_subparsers(help = 'Arguments for specific dataset types.', dest = 'dataset_type')
    subparsers.required = True

    linemod_parser = subparsers.add_parser('linemod')
    linemod_parser.add_argument('linemod_path', help = 'Path to dataset directory (ie. /Datasets/Linemod_preprocessed).')
    linemod_parser.add_argument('--object-id', help = 'ID of the Linemod Object to train on', type = int, default = 8)

    occlusion_parser = subparsers.add_parser('occlusion')
    occlusion_parser.add_argument('occlusion_path', help = 'Path to dataset directory (ie. /Datasets/Linemod_preprocessed/).')

    #parser.add_argument('--rotation-representation', help = 'Which representation of the rotation should be used. Choose from "axis_angle", "rotation_matrix" and "quaternion"', default = 'axis_angle')
    parser.add_argument('--rotation-representation',
                        help='Which representation of the rotation should be used. Choose from "axis_angle", "rotation_matrix", "quaternion" and "r_6d"',
                        default='r_6d')

    parser.add_argument('--weights', help = 'File containing weights to init the model parameter')
    parser.add_argument('--freeze-backbone', help = 'Freeze training of backbone layers.', action = 'store_false')
    parser.add_argument('--no-freeze-bn', help = 'Do not freeze training of BatchNormalization layers.', action = 'store_true')

    parser.add_argument('--batch-size', help = 'Size of the batches.', default = 8, type = int)
    parser.add_argument('--lr', help = 'Learning rate', default = 1e-4, type = float)
    parser.add_argument('--no-color-augmentation', help = 'Do not use colorspace augmentation', action = 'store_true')
    parser.add_argument('--no-6dof-augmentation', help = 'Do not use 6DoF augmentation', action = 'store_true')
    #phi不同参数代表不同网络、
    parser.add_argument('--phi', help = 'Hyper parameter phi', default = 0, type = int, choices = (0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', default="3", help = 'Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs', help = 'Number of epochs to train.', type = int, default = 200)
    parser.add_argument('--steps', help = 'Number of steps per epoch.', type = int, default = int(600))
    parser.add_argument('--snapshot-path', help = 'Path to store snapshots of models during training', default = os.path.join("check", date_and_time))
    parser.add_argument('--tensorboard-dir', help = 'Log directory for Tensorboard output', default = os.path.join("logs", date_and_time))
    parser.add_argument('--no-snapshots', help = 'Disable saving snapshots.', dest = 'snapshots', action = 'store_false')
    parser.add_argument('--no-evaluation', help = 'Disable per epoch evaluation.', dest = 'evaluation', action = 'store_false')
    parser.add_argument('--compute-val-loss', help = 'Compute validation loss during training', dest = 'compute_val_loss', action = 'store_true')
    parser.add_argument('--score-threshold', help = 'score threshold for non max suppresion', type = float, default = 0.5)
    parser.add_argument('--validation-image-save-path', help = 'path where to save the predicted validation images after each epoch', default = None)

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help = 'Use multiprocessing in fit_generator.', action = 'store_true')
    parser.add_argument('--workers', help = 'Number of generator workers.', type = int, default = 32)
    parser.add_argument('--max-queue-size', help = 'Queue length for multiprocessing workers in fit_generator.', type = int, default = 10)
    
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def main(args = None):
    """
    Train an model.

    Args:
        args: parseargs object containing configuration for the training procedure.
    """
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF
    sess = tf.Session(config=config)
      
    allow_gpu_growth_memory()
    
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
      
    # create the generators
    print("\nCreating the Generators...")
    train_generator, validation_generator = create_generators(args)

    num_rotation_parameters = train_generator.get_num_rotation_parameters()
    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    # optionally choose specific GPU
    if args.gpu:
        print("Selected GPU:", args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("\nBuilding the Model...")
    model, prediction_model, all_layers = build_Net(args.phi,
                                                    num_classes = num_classes,
                                                    num_anchors = num_anchors,
                                                    freeze_bn = not args.no_freeze_bn,
                                                    score_threshold = args.score_threshold,
                                                    num_rotation_parameters = num_rotation_parameters)
    print("Done!")
    if args.weights:
        print('Loading model, this may take a second...')
        custom_load_weights(filepath = args.weights, layers = all_layers, skip_mismatch = True)
        print("\nDone!")

    print(str(args.freeze_backbone))


    model.compile(optimizer=Adam(lr = args.lr, clipnorm = 0.001), 
                  loss={'regression': smooth_l1(),
                        'classification': focal(),
                        'transformation': transformation_rot_loss(model_3d_points_np = train_generator.get_all_3d_model_points_array_for_loss(),
                                                              num_rotation_parameter = num_rotation_parameters)},
                  loss_weights = {'regression' : 1.0,
                                  'classification': 1.0,
                                  'transformation': 0.04})#0.02

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None
    elif args.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    # start training
    return model.fit_generator(
        generator = train_generator,
        steps_per_epoch = args.steps,
        initial_epoch = 0,
        epochs = args.epochs,
        verbose = 1,
        callbacks = callbacks,
        workers = args.workers,
        use_multiprocessing = args.multiprocessing,
        max_queue_size = args.max_queue_size,
        validation_data = validation_generator
    )


def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)


def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args:
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None
    
    if args.dataset_type == "linemod":
        snapshot_path = os.path.join(args.snapshot_path, "object_" + str(args.object_id))
        if args.validation_image_save_path:
            save_path = os.path.join(args.validation_image_save_path, "object_" + str(args.object_id))
        else:
            save_path = args.validation_image_save_path
        if args.tensorboard_dir:
            tensorboard_dir = os.path.join(args.tensorboard_dir, "object_" + str(args.object_id))  #logs/time/obj_文件夹
            
        if validation_generator.is_symmetric_object(args.object_id):
            metric_to_monitor = "ADD-S"
            mode = "max"
        else:
            metric_to_monitor = "ADD"
            mode = "max"
    elif args.dataset_type == "occlusion":
        snapshot_path = os.path.join(args.snapshot_path, "occlusion")
        if args.validation_image_save_path:
            save_path = os.path.join(args.validation_image_save_path, "occlusion")
        else:
            save_path = args.validation_image_save_path
        if args.tensorboard_dir:
            tensorboard_dir = os.path.join(args.tensorboard_dir, "occlusion")
            
        metric_to_monitor = "ADD(-S)"
        mode = "max"
    else:
        snapshot_path = args.snapshot_path
        save_path = args.validation_image_save_path
        tensorboard_dir = args.tensorboard_dir
        
    if save_path:
        os.makedirs(save_path, exist_ok = True)

    if tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir = tensorboard_dir,
            histogram_freq = 0,
            batch_size = args.batch_size,
            write_graph = True,
            write_grads = False,
            write_images = False,
            embeddings_freq = 0,
            embeddings_layer_names = None,
            embeddings_metadata = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        from eval.eval_callback import Evaluate
        evaluation = Evaluate(validation_generator, prediction_model, tensorboard = tensorboard_callback, save_path = save_path)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(snapshot_path, exist_ok = True)
        filepath = os.path.join(snapshot_path, 'phi_{phi}_{dataset_type}_best_{metric}.h5'.format(phi = str(args.phi), metric = metric_to_monitor, dataset_type = args.dataset_type))

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath,
            verbose=1,
            save_best_only=True,
            monitor=metric_to_monitor,
            mode=mode,
            save_format='tf'
        )
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'MixedAveragePointDistanceMean_in_mm',
        factor     = 0.5,
        patience   = 20,
        verbose    = 1,
        mode       = 'min',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 1e-7
    ))

    return callbacks


def create_generators(args):
    """
    Create generators for training and validation.

    Args:
        args: parseargs object containing configuration for generators.
    Returns:
        The training and validation generators.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }

    if args.dataset_type == 'linemod':
        from generators.linemod import LineModGenerator
        train_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )
        validation_generator = LineModGenerator(
            args.linemod_path,
            args.object_id,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    elif args.dataset_type == 'occlusion':
        from generators.occlusion import OcclusionGenerator
        #
        train_generator = OcclusionGenerator(
            args.occlusion_path,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = not args.no_color_augmentation,
            use_6DoF_augmentation = not args.no_6dof_augmentation,
            **common_args
        )

        validation_generator = OcclusionGenerator(
            args.occlusion_path,
            train = False,
            shuffle_dataset = False,
            shuffle_groups = False,
            rotation_representation = args.rotation_representation,
            use_colorspace_augmentation = False,
            use_6DoF_augmentation = False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


if __name__ == '__main__':
    main()
