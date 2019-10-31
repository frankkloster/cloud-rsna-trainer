import keras

from tensorflow.python.lib.io import file_io

import numpy as np
import pandas as pd

import pydicom
import cv2

import gcsfs

from math import ceil
import os

TEST_IMAGES_DIR = os.environ.get('test_images_dir')
TRAIN_IMAGES_DIR = os.environ.get('train_images_dir')


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img


def bsb_window(dcm):
    """
    Brain-Subdural-Bone Windowing.
    """
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return bsb_img


def window_with_correction(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def window_without_correction(dcm, window_center, window_width):
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def window_testing(img, window):
    brain_img = window(img, 40, 80)
    subdural_img = window(img, 80, 200)
    soft_img = window(img, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return bsb_img


def _read(path, desired_size):
    """Will be used in DataGenerator"""
    if path.startswith('gs://'):
        project = 'imperial-legacy-197723'
        try:
            fs = gcsfs.GCSFileSystem(project=project)
            with fs.open(path) as f:
                dcm = pydicom.dcmread(f)
        except Exception as e:
            print(e)
            raise
    else:
        dcm = pydicom.dcmread(path)

    try:
        img = bsb_window(dcm)
    except:
        img = np.zeros(desired_size)
    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)
    return img


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=(512, 512, 1),
                 img_dir=TRAIN_IMAGES_DIR, *args, **kwargs):
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        if img_dir[-1] != '/':
            img_dir += '/'
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))

        if self.labels is not None:  # training phase
            Y = np.empty((self.batch_size, 6), dtype=np.float32)

            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir + ID + ".dcm", self.img_size)
                Y[i,] = self.labels.loc[ID].values

            return X, Y

        else:  # test phase
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir + ID + ".dcm", self.img_size)

            return X


# h5py workaround: copy local models over to GCS if the job_dir is GCS.
def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(
                os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def read_testset(filename='gs://rsna-kaggle-data/csv/stage_1_sample_submission.csv'):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df


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