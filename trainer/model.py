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
"""Implements the Keras Sequential model."""

from builtins import range

import keras

from keras import backend as K
from keras import layers
from keras import models
from keras.backend import relu

import pandas as pd
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from trainer.data import DataGenerator
from trainer.checkpoints import PredictionCheckpoint

import os

TEST_IMAGES_DIR = os.environ.get('test_images_dir')
TRAIN_IMAGES_DIR = os.environ.get('train_images_dir')

# CSV columns in the input file.
CSV_COLUMNS = ('Image')

CSV_COLUMN_DEFAULTS = ['']

LABELS = [0, 1]
LABEL_COLUMNS = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


class MyDeepModel:
    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3,
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = weights
        self.verbose = verbose
        self._build()

    def _build(self):
        engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims,
                             backend=keras.backend, layers=keras.layers,
                             models=keras.models, utils=keras.utils)

        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)

        self.model = keras.models.Model(inputs=engine.input, outputs=out)

        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])

    def fit_and_predict(self, train_df, valid_df, test_df):
        # callbacks
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps))
        )

        self.model.fit_generator(
            DataGenerator(
                train_df.index,
                train_df,
                self.batch_size,
                self.input_dims,
                TRAIN_IMAGES_DIR
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=4,
            callbacks=[pred_history, scheduler]
        )

        return pred_history

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

def model_fn(input_dim,
             labels_dim,
             hidden_units=[100, 70, 50, 20],
             learning_rate=0.1):
  """Create a Keras Sequential model with layers.
  Args:
    input_dim: (int) Input dimensions for input layer.
    labels_dim: (int) Label dimensions for input layer.
    hidden_units: [int] the layer sizes of the DNN (input layer first)
    learning_rate: (float) the learning rate for the optimizer.
  Returns:
    A Keras model.
  """

  # "set_learning_phase" to False to avoid:
  # AbortionError(code=StatusCode.INVALID_ARGUMENT during online prediction.
  K.set_learning_phase(False)
  model = models.Sequential()

  for units in hidden_units:
    model.add(layers.Dense(units=units, input_dim=input_dim, activation=relu))
    input_dim = units

  # Add a dense final layer with sigmoid function.
  model.add(layers.Dense(labels_dim, activation='sigmoid'))
  compile_model(model, learning_rate)
  return model


def compile_model(model, learning_rate):
  model.compile(
      loss='binary_crossentropy',
      optimizer=keras.optimizers.Adam(lr=learning_rate),
      metrics=['accuracy'])
  return model


def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(
      inputs={'input': model.inputs[0]}, outputs={'income': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        })
    builder.save()


def generator_input(filenames, chunk_size, batch_size=64):
  """Produce features and labels needed by keras fit_generator."""

  feature_cols = None
  while True:
    input_reader = pd.read_csv(
        tf.gfile.Open(filenames[0]),
        names=CSV_COLUMNS,
        chunksize=chunk_size,
        na_values=' ?')

    for input_data in input_reader:
      input_data = input_data.dropna()
      label = pd.get_dummies(input_data.pop(LABEL_COLUMN))

      input_data = to_numeric_features(input_data, feature_cols)

      # Retains schema for next chunk processing.
      if feature_cols is None:
        feature_cols = input_data.columns

      idx_len = input_data.shape[0]
      for index in range(0, idx_len, batch_size):
        yield (input_data.iloc[index:min(idx_len, index + batch_size)],
               label.iloc[index:min(idx_len, index + batch_size)])