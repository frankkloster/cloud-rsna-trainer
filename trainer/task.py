# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This code implements a Feed forward neural network using Keras API."""

import argparse
import os

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras_applications.resnet import ResNet50

from tensorflow.python.lib.io import file_io

import trainer.model as model
from trainer.callbacks import ContinuousEval
from trainer.data import read_trainset


# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = 5000
CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
CENSUS_MODEL = 'rsna.hdf5'

ENGINE = ResNet50
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

    # Model checkpoint callback.
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=args.checkpoint_epochs,
        mode='min')

    # Continuous eval callback.
    evaluation = ContinuousEval(args.eval_frequency, args.eval_files,
                                args.learning_rate, args.job_dir)

    # Tensorboard logs callback.
    tb_log = TensorBoard(
        log_dir=os.path.join(args.job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, evaluation, tb_log]

    # Get the data
    train_df = read_trainset()

    cnn_model.fit_and_predict(train_df.iloc[train_idx], train_df.iloc[valid_idx], test_df, callbacks)

    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    if args.job_dir.startswith('gs://'):
        cnn_model.save(CENSUS_MODEL)
        copy_file_to_gcs(args.job_dir, CENSUS_MODEL)
    else:
        cnn_model.save(os.path.join(args.job_dir, CENSUS_MODEL))

    # Convert the Keras model to TensorFlow SavedModel.
    model.to_savedmodel(cnn_model, os.path.join(args.job_dir, 'export'))


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
                os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-files',
        nargs='+',
        help='Training file local or GCS',
        default=['gs://cloud-samples-data/ml-engine/census/data/adult.data.csv'])
    parser.add_argument(
        '--eval-files',
        nargs='+',
        help='Evaluation file local or GCS',
        default=['gs://cloud-samples-data/ml-engine/census/data/adult.test.csv'])
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS or local dir to write checkpoints and export model',
        default='/tmp/census-keras')
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
        '--first-layer-size',
        type=int,
        default=256,
        help='Number of nodes in the first layer of DNN')
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of layers in DNN')
    parser.add_argument(
        '--scale-factor',
        type=float,
        default=0.25,
        help="""Rate of decay size of layer for Deep Neural Net.
        max(2, int(first_layer_size * scale_factor**i))""")
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
