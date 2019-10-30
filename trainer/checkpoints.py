import keras
from trainer.data import DataGenerator
import os

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