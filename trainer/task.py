import argparse
import os
from math import floor

from sklearn.model_selection import ShuffleSplit

import trainer.model as model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras_applications.inception_v3 import InceptionV3
from trainer.data import copy_file_to_gcs

# from trainer.callbacks import ContinuousEval

CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
MODEL_FILE = 'rsna.hdf5'

ENGINE = InceptionV3
INPUT_DIMS = (256, 256, 3)

"""
TODO:
So, this part is unneeded at this very moment. To make a submission, it is needed. Maybe split it into two files, task-train and task-predict for the sample prediction?
"""
# def read_testset(filename='gs://rsna-kaggle-data/csv/stage_1_sample_submission.csv'):
#     df = pd.read_csv(filename)
#     df["Image"] = df["ID"].str.slice(stop=12)
#     df["Diagnosis"] = df["ID"].str.slice(start=13)

#     df = df.loc[:, ["Label", "Diagnosis", "Image"]]
#     df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

#     return df


def read_trainset(filename="gs://rsna-kaggle-data/csv/stage_1_train.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    duplicates_to_remove = [
        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
        312468, 312469, 312470, 312471, 312472, 312473,
        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
        3032994, 3032995, 3032996, 3032997, 3032998, 3032999
    ]

    df = df.drop(index=duplicates_to_remove)
    df = df.reset_index(drop=True)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df

def train_and_evaluate(args):
    cnn_model = model.MyDeepModel(
        engine=ENGINE, 
        input_dims=INPUT_DIMS,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        weights='imagenet',
        verbose=1
    )
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
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training steps')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=32,
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
        default=5,
        help='Maximum number of epochs on which to train')
    parser.add_argument(
        '--checkpoint-epochs',
        type=int,
        default=5,
        help='Checkpoint per n training epochs')

    args, _ = parser.parse_known_args()
    train_and_evaluate(args)
