import keras
from keras.models import load_model
from keras.callbacks import Callback

from trainer.data import DataGenerator, copy_file_to_gcs

import os
import glob

import trainer.model as model

TEST_IMAGES_DIR = os.environ.get('test_images_dir')
TRAIN_IMAGES_DIR = os.environ.get('train_images_dir')


class PredictionCheckpoint(keras.callbacks.Callback):
    def __init__(self, test_df, valid_df,
                 test_images_dir=TEST_IMAGES_DIR,
                 valid_images_dir=TRAIN_IMAGES_DIR,
                 batch_size=32, input_size=(224, 224, 3)):
        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = valid_images_dir
        self.batch_size = batch_size
        self.input_size = input_size

    def on_train_begin(self, logs={}):
        self.test_predictions = []
        self.valid_predictions = []

    def on_epoch_end(self, batch, logs={}):
        self.test_predictions.append(
            self.model.predict_generator(
                DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, self.test_images_dir),
                verbose=2)[:len(self.test_df)]
        )


class ContinuousEval(Callback):
    """Continuous eval callback to evaluate the checkpoint once
     every so many epochs.
  """
    def __init__(self,
                 eval_frequency,
                 eval_files,
                 learning_rate,
                 job_dir,
                 steps=1000):
        self.eval_files = eval_files
        self.eval_frequency = eval_frequency
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.steps = steps

    def on_epoch_begin(self, epoch, logs={}):
        """Compile and save model."""
        if epoch > 0 and epoch % self.eval_frequency == 0:
            # Unhappy hack to work around h5py not being able to write to GCS.
            # Force snapshots and saves to local filesystem, then copy them over to GCS.
            model_path_glob = 'checkpoint.*'
            if not self.job_dir.startswith('gs://'):
                model_path_glob = os.path.join(self.job_dir, model_path_glob)
            checkpoints = glob.glob(model_path_glob)
            if len(checkpoints) > 0:
                checkpoints.sort()
                census_model = load_model(checkpoints[-1])
                census_model = model.compile_model(census_model, self.learning_rate)
                loss, acc = census_model.evaluate_generator(
                    model.generator_input(self.eval_files, chunk_size=CHUNK_SIZE),
                    steps=self.steps)
                print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
                    epoch, loss, acc, census_model.metrics_names))
                if self.job_dir.startswith('gs://'):
                    copy_file_to_gcs(self.job_dir, checkpoints[-1])
            else:
                print('\nEvaluation epoch[{}] (no checkpoints found)'.format(epoch))