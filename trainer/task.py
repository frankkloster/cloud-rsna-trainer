import argparse
import os
from math import floor

from sklearn.model_selection import ShuffleSplit

import trainer.model as model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras_applications.inception_v3 import InceptionV3
from trainer.data import copy_file_to_gcs, read_testset, read_trainset

# from trainer.callbacks import ContinuousEval

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = 5000
CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
MODEL_FILE = 'rsna.hdf5'

ENGINE = InceptionV3
INPUT_DIMS = (256, 256, 3)


def train_and_evaluate(args):
    cnn_model = model.MyDeepModel(engine=ENGINE, input_dims=INPUT_DIMS)
    try:
        os.makedirs(args.job_dir)
    except:
        pass

    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = CHECKPOINT_FILE_PATH
    if not args.job_dir.startswith('gs://'):
        checkpoint_path = os.path.join(args.job_dir, checkpoint_path)

    # Get the data
    train_df = read_trainset()
    test_df = read_testset()

    # train set (00%) and validation set (10%)
    ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(train_df.index)

    # lets go for the first fold only
    train_idx, valid_idx = next(ss)

    # ===== CALLBACKS ======
    # Model checkpoint callback.
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=args.checkpoint_epochs,
        mode='min')

    # Continuous eval callback.
    # evaluation = ContinuousEval(args.eval_frequency, args.eval_files,
    #                             args.learning_rate, args.job_dir)

    # Tensorboard logs callback.
    tb_log = TensorBoard(
        log_dir=os.path.join(args.job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    scheduler = LearningRateScheduler(
        lambda epoch: args.learning_rate * pow(args.decay_rate, floor(epoch / args.decay_steps))
    )

    callbacks = [checkpoint, tb_log, scheduler]

    cnn_model.fit_and_predict(train_df.iloc[train_idx], train_df.iloc[valid_idx], test_df, callbacks,
                              args.train_images_dir)

    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if args.job_dir.startswith('gs://'):
        cnn_model.save(MODEL_FILE)
        copy_file_to_gcs(args.job_dir, MODEL_FILE)
    else:
        cnn_model.save(os.path.join(args.job_dir, MODEL_FILE))

    # Convert the Keras model to TensorFlow SavedModel.
    model.to_savedmodel(cnn_model, os.path.join(args.job_dir, 'export'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-images-dir',
        nargs='+',
        help='Training images local or GCS',
        default='gs://rsna-kaggle-data/images/stage_1_train_images')
    parser.add_argument(
        '--test-images-dir',
        nargs='+',
        help='Testing images local or GCS',
        default='gs://rsna-kaggle-data/images/stage_1_test_images')
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default='gs://gcc-models/rsna')
    parser.add_argument(
        '--train-steps',
        type=int,
        default=100,
        help="""\
        Maximum number of training steps to perform
        Training steps are in the units of training-batch-size.
        So if train-steps is 500 and train-batch-size if 100 then
        at most 500 * 100 training instances will be used to train.""")
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int)
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=40,
        help='Batch size for training steps')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=40,
        help='Batch size for evaluation steps')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.003,
        help='Learning rate for SGD')
    parser.add_argument(
        '--eval-frequency',
        default=10,
        help='Perform one evaluation per n epochs')
    parser.add_argument(
        '--scale-factor',
        type=float,
        default=0.25,
        help="""Rate of decay size of layer for Deep Neural Net.
        max(2, int(first_layer_size * scale_factor**i))""")
    parser.add_argument(
        '--decay-rate',
        type=float,
        default=1.0,
        help='Decay rate for learning rate schedule')
    parser.add_argument(
        '--decay-steps',
        type=int,
        default=1,
        help='Decay steps for learning rate schedule')
    parser.add_argument(
        '--eval-num-epochs',
        type=int,
        default=1,
        help='Number of epochs during evaluation')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='Maximum number of epochs on which to train')
    parser.add_argument(
        '--checkpoint-epochs',
        type=int,
        default=5,
        help='Checkpoint per n training epochs')

    args, _ = parser.parse_known_args()
    train_and_evaluate(args)
