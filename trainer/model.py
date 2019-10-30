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
from trainer.callbacks import PredictionCheckpoint

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from trainer.data import DataGenerator
from trainer.loss_eval_fcns import weighted_loss

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

    def fit_and_predict(self, train_df, valid_df, test_df, callbacks):
        # Prediction checkpoint
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        callbacks += pred_history

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
            callbacks=callbacks
        )

        return pred_history

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


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