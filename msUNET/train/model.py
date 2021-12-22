# Import packages

# Import Required ML Framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam as Adam
from tensorflow.keras.utils import multi_gpu_model as multi_gpu_model
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Legacy from previously using standalone Keras
#import keras
#from keras import models
#from keras.models import load_model
#from keras import backend as K
#from keras.optimizers import adam as Adam
#from tensorflow.keras.optimizers import Adam as Adam

import talos
from talos.model.normalizers import lr_normalizer

# Import data manipulation packages
import numpy as np
import pandas as pd

# Import image manipulation packages
from image import resample_img, min_max_normalization, image_patch, mask_patch, uniform_mask
import SimpleITK as sitk

# Import assorted essential metric and loss functions
from metrics import dice_coef, dice_coef_metric, dice_coef_loss, jaccard_similarity, jaccard_distance, dice_coef_anatomical, dice_coef_loss_by_sample
from util import layer_parse, save_object, remove_x_y_data, load_data
from augmentation import rotation_augmentation, affine_augmentation

# Import some various useful packages
import os
import argparse
import math
import time
import shutil
from zipfile import ZipFile
import pickle
import gc


# Define a callback class to profile metrics on a by-modality, per epoch basis
class modalityMetrics(keras.callbacks.Callback):
    # Keras callback class that profiles metrics on a by-modality, by-epoch basis.
    # Hard-coded to only support the four modalities most relevant in the initial work:
    # Anatomical, DTI, fMRI, NODDI.
    # Currently disabled, as performance implications can be grave.
    def __init__(self,
                 x_train,
                 y_train,
                 x_val,
                 y_val,
                 anatomical_val_indices,
                 dti_val_indices,
                 noddi_val_indices,
                 regwarp_val_indices,
                 fmri_val_indices,
                 anatomical_train_indices,
                 dti_train_indices,
                 noddi_train_indices,
                 regwarp_train_indices,
                 fmri_train_indices,
                 save_location,
                 name='modality_metrics'):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.anatomical_val_indices = anatomical_val_indices
        self.dti_val_indices = dti_val_indices
        self.noddi_val_indices = noddi_val_indices
        self.regwarp_val_indices = regwarp_val_indices
        self.fmri_val_indices = fmri_val_indices
        self.anatomical_train_indices = anatomical_train_indices
        self.dti_train_indices = dti_train_indices
        self.noddi_train_indices = noddi_train_indices
        self.regwarp_train_indices = regwarp_train_indices
        self.fmri_train_indices = fmri_train_indices
        self.save_location = save_location

    def on_epoch_end(self, epoch, logs=None):
        self.anatomical_val_performance = [None, None, None]
        self.dti_val_performance = [None, None, None]
        self.noddi_val_performance = [None, None, None]
        self.regwarp_val_performance = [None, None, None]
        self.fmri_val_performance = [None, None, None]
        self.anatomical_train_performance = [None, None, None]
        self.dti_train_performance = [None, None, None]
        self.noddi_train_performance = [None, None, None]
        self.regwarp_train_performance = [None, None, None]
        self.fmri_train_performance = [None, None, None]
        print('Begin by-epoch, by-modality metric calculation - Validation')
        if len(self.anatomical_val_indices) > 0:
            self.anatomical_val_performance = self.model.evaluate(
                self.x_val[self.anatomical_val_indices, :, :, :], self.y_val[self.anatomical_val_indices, :, :, :])
        if len(self.dti_val_indices) > 0:
            self.dti_val_performance = self.model.evaluate(
                self.x_val[self.dti_val_indices, :, :, :], self.y_val[self.dti_val_indices, :, :, :])
        if len(self.noddi_val_indices) > 0:
            self.noddi_val_performance = self.model.evaluate(
                self.x_val[self.noddi_val_indices, :, :, :], self.y_val[self.noddi_val_indices, :, :, :])
        if len(self.regwarp_val_indices) > 0:
            self.regwarp_val_performance = self.model.evaluate(
                self.x_val[self.regwarp_val_indices, :, :, :], self.y_val[self.regwarp_val_indices, :, :, :])
        if len(self.fmri_val_indices) > 0:
            self.fmri_val_performance = self.model.evaluate(
                self.x_val[self.fmri_val_indices, :, :, :], self.y_val[self.fmri_val_indices, :, :, :])
        print('Begin by-epoch, by-modality metric calculation - Training')
        if len(self.anatomical_train_indices) > 0:
            self.anatomical_train_performance = self.model.evaluate(
                self.x_train[self.anatomical_train_indices, :, :, :], self.y_train[self.anatomical_train_indices, :, :, :])
        if len(self.dti_train_indices) > 0:
            self.dti_train_performance = self.model.evaluate(
                self.x_train[self.dti_train_indices, :, :, :], self.y_train[self.dti_train_indices, :, :, :])
        if len(self.noddi_train_indices) > 0:
            self.noddi_train_performance = self.model.evaluate(
                self.x_train[self.noddi_train_indices, :, :, :], self.y_train[self.noddi_train_indices, :, :, :])
        if len(self.regwarp_train_indices) > 0:
            self.regwarp_train_performance = self.model.evaluate(
                self.x_train[self.regwarp_train_indices, :, :, :], self.y_train[self.regwarp_train_indices, :, :, :])
        if len(self.fmri_train_indices) > 0:
            self.fmri_train_performance = self.model.evaluate(
                self.x_train[self.fmri_train_indices, :, :, :], self.y_train[self.fmri_train_indices, :, :, :])

        # Change with commented code above to return to training on all modalities
        #self.dti_train_performance = self.model.evaluate(self.x_train,self.y_train)

        column_names = [
            'anatomical_val_dice',
            'dti_val_dice',
            'noddi_val_dice',
            'regwarp_val_dice',
            'fmri_val_dice',
            'anatomical_train_dice',
            'dti_train_dice',
            'noddi_train_dice',
            'regwarp_train_dice',
            'fmri_train_dice']
        df = pd.DataFrame(columns=column_names)
        df.loc[0] = [
            self.anatomical_val_performance[1],
            self.dti_val_performance[1],
            self.noddi_val_performance[1],
            self.regwarp_val_performance[1],
            self.fmri_val_performance[1],
            self.anatomical_train_performance[1],
            self.dti_train_performance[1],
            self.noddi_train_performance[1],
            self.regwarp_train_performance[1],
            self.fmri_train_performance[1]]
        df.to_csv(self.save_location, mode='a', header=False)


def mouseSegUNET(x_train, y_train, x_val, y_val, params):
    # Function that builds and trains the UNET model. Also supports doing PCA on the intermediate
    # representation if requested.
    # Define parallelization strategy
    #strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    # with strategy.scope():
    # Get all the information contained in x_train, x_val extracted into
    # helpful forms
    print('Model initialization flag')
    # Make the current patch size and stride into something of the form
    # size/stride
    pca_only = params['pca_only']
    if isinstance(params['patch_dimensions'], int):
        current_patch_dimension_1 = params['patch_dimensions']
        current_patch_dimension_2 = params['patch_dimensions']
    elif isinstance(params['patch_dimensions'], str):
        current_patch_dimension_1 = int(
            params['patch_dimensions'].split('/')[0])
        current_patch_dimension_2 = int(
            params['patch_dimensions'].split('/')[1])
        print(current_patch_dimension_1)
        print(current_patch_dimension_2)
    current_stride = params['patch_stride']
    current_dataset_label = str(
        current_patch_dimension_1) + '/' + str(current_stride)
    # Look through x_train elements to find a matching combination size/stride
    # Extract the next element, which will be the numpy array containing the
    # dataset
    current_x_train_dataset_index = x_train.index(current_dataset_label) + 2
    current_dataset_index = math.floor(current_x_train_dataset_index / 3)
    # Grab the validation dataset index and corresponding
    current_x_val_dataset_index = math.floor(current_x_train_dataset_index / 2)
    # Pull the lists corresponding to the modalities of each individual image
    # patch, training and validation
    train_modality_list = x_train[current_x_train_dataset_index - 1]
    validation_modality_list = x_val[current_x_val_dataset_index - 1]
    x_train_data = np.array(
        x_train[current_x_train_dataset_index],
        dtype=np.float32)
    y_train_data = np.array(y_train[current_dataset_index], dtype=np.float32)
    x_val_data = np.array(x_val[current_x_val_dataset_index], dtype=np.float32)
    y_val_data = np.array(y_val[current_dataset_index], dtype=np.float32)
    print(x_train_data.shape)
    print(y_train_data.shape)
    print(current_patch_dimension_1)
    print(current_patch_dimension_2)
    print(np.array(train_modality_list).shape)
    print(np.array(validation_modality_list).shape)

    # Grab the modality of each slice - training
    anatomical_train_indices = [
        index for index,
        element in enumerate(train_modality_list) if element == 0]
    dti_train_indices = [index for index, element in enumerate(
        train_modality_list) if element == 1]
    noddi_train_indices = [
        index for index,
        element in enumerate(train_modality_list) if element == 2]
    regwarp_train_indices = [
        index for index,
        element in enumerate(train_modality_list) if element == 3]
    fmri_train_indices = [index for index, element in enumerate(
        train_modality_list) if element == 4]

    # Grab the indicies of each modality - validation
    anatomical_val_indices = [index for index, element in enumerate(
        validation_modality_list) if element == 0]
    dti_val_indices = [index for index, element in enumerate(
        validation_modality_list) if element == 1]
    noddi_val_indices = [index for index, element in enumerate(
        validation_modality_list) if element == 2]
    regwarp_val_indices = [index for index, element in enumerate(
        validation_modality_list) if element == 3]
    fmri_val_indices = [index for index, element in enumerate(
        validation_modality_list) if element == 4]

    if params['modality_weight'] is not None:
        train_sample_weight = np.ones(shape=(len(train_modality_list),))
        validation_sample_weight = np.ones(
            shape=(len(validation_modality_list),))
        if params['modality_weight'] == 'anatomical':
            print('Weighting anatomical training samples a factor of ' +
                  str(params['weight_factor']) + ' higher.')
            train_sample_weight[anatomical_train_indices] = params['weight_factor']
            validation_sample_weight[anatomical_val_indices] = params['weight_factor']
        if params['modality_weight'] == 'dti':
            print('Weighting dti training samples a factor of ' +
                  str(params['weight_factor']) + ' higher.')
            train_sample_weight[dti_train_indices] = train_sample_weight[dti_train_indices] * \
                params['weight_factor']
            validation_sample_weight[dti_val_indices] = validation_sample_weight[dti_val_indices] * \
                params['weight_factor']
        if params['modality_weight'] == 'fmri':
            print('Weighting fmri training samples a factor of ' +
                  str(params['weight_factor']) + ' higher.')
            train_sample_weight[fmri_train_indices] = train_sample_weight[fmri_train_indices] * \
                params['weight_factor']
            validation_sample_weight[fmri_val_indices] = validation_sample_weight[fmri_val_indices] * \
                params['weight_factor']
        if params['modality_weight'] == 'noddi':
            print('Weighting noddi training samples a factor of ' +
                  str(params['weight_factor']) + ' higher.')
            train_sample_weight[noddi_train_indices] = params['weight_factor']
            validation_sample_weight[noddi_val_indices] = validation_sample_weight[noddi_val_indices] * \
                params['weight_factor']
        sample_weight = np.array(train_sample_weight)
        print(np.amax(sample_weight))
        print(sample_weight.shape)
    # Cut the validation data, as we only want to do PCA on a single modality
    #x_val_data = x_val_data[anatomical_val_indices]

    print('Datasets loaded, moving to model definition')
    inputLayer = keras.Input(
        shape=(
            current_patch_dimension_1,
            current_patch_dimension_2,
            1),
        name='input_1')
    conv1_1 = keras.layers.Conv2D(
        filters=32,
        name='conv1_1',
        trainable=(
            True if params['which_layer'] > 1 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')
    print('the first layer is trainable ' +
          str(True if params['which_layer'] > 1 else False))
    conv1_1out = conv1_1(inputLayer)
    conv1_1bnout = keras.layers.BatchNormalization(
        name='conv1_1_bn', trainable=params['expandingBatchNormTrainable'])(conv1_1out)
    conv1_2out = keras.layers.Conv2D(
        filters=32,
        name='conv1_2',
        trainable=(
            True if params['which_layer'] > 2 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv1_1bnout)
    conv1_2bnout = keras.layers.BatchNormalization(
        name='conv1_2_bn', trainable=params['expandingBatchNormTrainable'])(conv1_2out)
    maxPool1_out = keras.layers.MaxPooling2D(
        pool_size=(
            2, 2), name='pool1', strides=(
            2, 2), padding='valid', data_format='channels_last')(conv1_2bnout)
    conv2_1out = keras.layers.Conv2D(
        filters=64,
        name='conv2_1',
        trainable=(
            True if params['which_layer'] > 3 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(maxPool1_out)
    conv2_1bnout = keras.layers.BatchNormalization(
        name='conv2_1_bn', trainable=params['expandingBatchNormTrainable'])(conv2_1out)
    conv2_2out = keras.layers.Conv2D(
        filters=64,
        name='conv2_2',
        trainable=(
            True if params['which_layer'] > 4 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv2_1bnout)
    conv2_2bnout = keras.layers.BatchNormalization(
        name='conv2_2_bn', trainable=params['expandingBatchNormTrainable'])(conv2_2out)
    maxPool2_out = keras.layers.MaxPooling2D(
        pool_size=(
            2, 2), name='pool2', strides=(
            2, 2), padding='valid', data_format='channels_last')(conv2_2bnout)
    conv3_1out = keras.layers.Conv2D(
        filters=96,
        name='conv3_1',
        trainable=(
            True if params['which_layer'] > 5 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(maxPool2_out)
    conv3_1bnout = keras.layers.BatchNormalization(
        name='conv3_1_bn', trainable=params['expandingBatchNormTrainable'])(conv3_1out)
    conv3_2out = keras.layers.Conv2D(
        filters=96,
        name='conv_3_2',
        trainable=(
            True if params['which_layer'] > 6 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv3_1bnout)
    conv3_2bnout = keras.layers.BatchNormalization(
        name='conv3_2_bn', trainable=params['expandingBatchNormTrainable'])(conv3_2out)
    maxPool3_out = keras.layers.MaxPooling2D(
        pool_size=(
            2, 2), name='pool3', strides=(
            2, 2), padding='valid', data_format='channels_last')(conv3_2bnout)
    conv4_1out = keras.layers.Conv2D(
        filters=128,
        name='conv4_1',
        trainable=(
            True if params['which_layer'] > 7 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(maxPool3_out)
    conv4_1bnout = keras.layers.BatchNormalization(
        name='conv4_1_bn', trainable=params['expandingBatchNormTrainable'])(conv4_1out)
    conv4_2out = keras.layers.Conv2D(
        filters=128,
        name='conv4_2',
        trainable=(
            True if params['which_layer'] > 8 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv4_1bnout)
    conv4_2bnout = keras.layers.BatchNormalization(
        name='conv4_2_bn', trainable=params['expandingBatchNormTrainable'])(conv4_2out)
    maxPool4_out = keras.layers.MaxPooling2D(
        pool_size=(
            2, 2), name='pool4', strides=(
            2, 2), padding='valid', data_format='channels_last')(conv4_2bnout)
    conv5_1out = keras.layers.Conv2D(
        filters=256,
        name='conv5_1',
        trainable=(
            True if params['which_layer'] > 9 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(maxPool4_out)
    conv5_1bnout = keras.layers.BatchNormalization(
        name='conv5_1_bn', trainable=params['expandingBatchNormTrainable'])(conv5_1out)
    conv5_2out = keras.layers.Conv2D(
        filters=256,
        name='conv5_2',
        trainable=(
            True if params['which_layer'] > 10 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv5_1bnout)
    conv5_2bnout = keras.layers.BatchNormalization(
        name='conv5_2_bn', trainable=params['expandingBatchNormTrainable'])(conv5_2out)
    up6_out = keras.layers.UpSampling2D(size=(2, 2),
                                        name='up6_c',
                                        data_format='channels_last',
                                        interpolation='nearest')(conv5_2bnout)
    up6_c = keras.layers.Conv2D(
        filters=128,
        name='up6',
        trainable=params['upsamplingTrainable'],
        kernel_size=(
            2,
            2),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_in',
            distribution='truncated_normal',
            seed=None),
        bias_initializer='zeros')(up6_out)
    up6_cbn = keras.layers.BatchNormalization(
        name='up6_bn', trainable=params['upsamplingBatchNormTrainable'])(up6_c)
    merge6_out = keras.layers.Concatenate(
        axis=-1, name='merge6')([conv4_2bnout, up6_cbn])
    drop6_out = keras.layers.Dropout(rate=0.5,
                                     noise_shape=None,
                                     seed=None,
                                     name='drop6')(merge6_out)
    conv6_1out = keras.layers.Conv2D(
        filters=128,
        name='conv6_1',
        trainable=(
            True if params['which_layer'] > 11 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(drop6_out)
    conv6_1bnout = keras.layers.BatchNormalization(
        name='conv6_1_bn',
        trainable=params['contractingBatchNormTrainable'])(conv6_1out)
    conv6_2out = keras.layers.Conv2D(
        filters=128,
        name='conv6_2',
        trainable=(
            True if params['which_layer'] > 12 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv6_1bnout)
    conv6_2bnout = keras.layers.BatchNormalization(
        name='conv6_2_bn',
        trainable=params['contractingBatchNormTrainable'])(conv6_2out)
    up7_out = keras.layers.UpSampling2D(size=(2, 2),
                                        name='up7_c',
                                        data_format='channels_last',
                                        interpolation='nearest')(conv6_2bnout)
    up7_c = keras.layers.Conv2D(
        filters=96,
        name='up7',
        trainable=params['upsamplingTrainable'],
        kernel_size=(
            2,
            2),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_in',
            distribution='truncated_normal',
            seed=None),
        bias_initializer='zeros')(up7_out)
    up7_cbn = keras.layers.BatchNormalization(
        name='up7_bn', trainable=params['upsamplingBatchNormTrainable'])(up7_c)
    merge7_out = keras.layers.Concatenate(
        axis=-1, name='merge7')([conv3_2bnout, up7_cbn])
    drop7_out = keras.layers.Dropout(rate=0.5,
                                     noise_shape=None,
                                     seed=None,
                                     name='drop7')(merge7_out)
    conv7_1out = keras.layers.Conv2D(
        filters=96,
        name='conv7_1',
        trainable=(
            True if params['which_layer'] > 13 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(drop7_out)
    conv7_1bnout = keras.layers.BatchNormalization(
        name='conv7_1_bn',
        trainable=params['contractingBatchNormTrainable'])(conv7_1out)
    conv7_2out = keras.layers.Conv2D(
        filters=96,
        name='conv7_2',
        trainable=(
            True if params['which_layer'] > 14 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv7_1bnout)
    conv7_2bnout = keras.layers.BatchNormalization(
        name='conv7_2_bn',
        trainable=params['contractingBatchNormTrainable'])(conv7_2out)
    up8_out = keras.layers.UpSampling2D(size=(2, 2),
                                        name='up8_c',
                                        data_format='channels_last',
                                        interpolation='nearest')(conv7_2bnout)
    up8_c = keras.layers.Conv2D(
        filters=64,
        name='up8',
        trainable=params['upsamplingTrainable'],
        kernel_size=(
            2,
            2),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_in',
            distribution='truncated_normal',
            seed=None),
        bias_initializer='zeros')(up8_out)
    up8_cbn = keras.layers.BatchNormalization(
        name='up8_bn', trainable=params['contractingBatchNormTrainable'])(up8_c)
    merge8_out = keras.layers.Concatenate(
        axis=-1, name='merge8')([conv2_2bnout, up8_cbn])
    drop8_out = keras.layers.Dropout(rate=0.5,
                                     name='drop8',
                                     noise_shape=None,
                                     seed=None)(merge8_out)
    conv8_1out = keras.layers.Conv2D(
        filters=64,
        name='conv8_1',
        trainable=(
            True if params['which_layer'] > 15 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(drop8_out)
    conv8_1bnout = keras.layers.BatchNormalization(
        name='conv8_1_bn',
        trainable=params['contractingBatchNormTrainable'])(conv8_1out)
    conv8_2out = keras.layers.Conv2D(
        filters=64,
        name='conv8_2',
        trainable=(
            True if params['which_layer'] > 16 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv8_1bnout)
    conv8_2bnout = keras.layers.BatchNormalization(
        name='conv8_2_bn',
        trainable=params['contractingBatchNormTrainable'])(conv8_2out)
    up9_out = keras.layers.UpSampling2D(size=(2, 2),
                                        name='up9_c',
                                        data_format='channels_last',
                                        interpolation='nearest')(conv8_2bnout)
    up9_c = keras.layers.Conv2D(
        filters=32,
        name='up9',
        trainable=params['upsamplingTrainable'],
        kernel_size=(
            2,
            2),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_in',
            distribution='truncated_normal',
            seed=None),
        bias_initializer='zeros')(up9_out)
    up9_cbn = keras.layers.BatchNormalization(
        name='up9_bn', trainable=params['contractingBatchNormTrainable'])(up9_c)
    merge9_out = keras.layers.Concatenate(
        axis=-1, name='merge9')([conv1_2bnout, up9_cbn])
    drop9_out = keras.layers.Dropout(rate=0.5,
                                     name='drop9',
                                     noise_shape=None,
                                     seed=None)(merge9_out)
    conv9_1out = keras.layers.Conv2D(
        filters=32,
        name='conv9_1',
        trainable=(
            True if params['which_layer'] > 17 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(drop9_out)
    conv9_1bnout = keras.layers.BatchNormalization(
        name='conv9_1_bn',
        trainable=params['contractingBatchNormTrainable'])(conv9_1out)
    conv9_2out = keras.layers.Conv2D(
        filters=32,
        name='conv9_2',
        trainable=(
            True if params['which_layer'] > 18 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv9_1bnout)
    conv9_2bnout = keras.layers.BatchNormalization(
        name='conv9_2_bn',
        trainable=params['contractingBatchNormTrainable'])(conv9_2out)
    conv9_3out = keras.layers.Conv2D(
        filters=2,
        name='conv9_3',
        trainable=(
            True if params['which_layer'] > 19 else False),
        kernel_size=(
            3,
            3),
        padding='same',
        data_format='channels_last',
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0,
                mode='fan_in',
                distribution='truncated_normal',
                seed=None),
        bias_initializer='zeros')(conv9_2bnout)
    score_out = keras.layers.Conv2D(
        filters=1,
        name='score',
        trainable=params['scoreTrainable'],
        kernel_size=(
            1,
            1),
        padding='same',
        data_format='channels_last',
        activation=params['finalActivation'],
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_in',
            distribution='truncated_normal',
            seed=None),
        bias_initializer='zeros')(conv9_3out)
    print('Model layers defined. Combining in keras.model')
    model = keras.Model(inputs=inputLayer,
                        outputs=score_out,
                        name='mouseSegUNET')
    # Check if we have more than 1 GPU available. If so, convert the base cpu
    # model to a multi GPU model
    print('Model combined. Moving to model compile')

    # Tell everybody where files should be saved for a given talos scan
    history_save_location = '/' + str(params['experiment_name'])
    history_save_location = history_save_location.replace('[', '')
    history_save_location = history_save_location.replace(']', '')
    history_save_location = history_save_location.replace('\'', '')
    history_save_location = history_save_location.replace('/', '')

    if pca_only:
        # Create intermediate model for principal component analysis of
        # 'center' layer of UNET
        intermediate_model = keras.Model(inputs=inputLayer,
                                         outputs=conv5_2bnout,
                                         name='intermediate_model')
        intermediate_model.compile(loss=dice_coef_loss,
                                   optimizer=Adam(learning_rate=params['lr']),
                                   metrics=[dice_coef, jaccard_similarity])
        intermediate_prediction = intermediate_model.predict(x_val_data)
        num_samples = intermediate_prediction.shape[0]
        intermediate_prediction = intermediate_prediction.flatten().reshape(num_samples, 16384)
        print('Built intermediate model, moving to pca fit')
        num_comp = 300
        pca = PCA(num_comp)
        projected = pca.fit_transform(intermediate_prediction)
        pca_var_plot = PCA().fit(intermediate_prediction)
        print('PCA fit complete')
        cum_var_fig = plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo')
        plt.title('Principle Component Analysis - MRI Segmentation')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.savefig(
            history_save_location +
            '/' +
            'pca_full_range.png',
            format='png')
        high_cum_var_fig = plt.figure()
        plt.plot(
            np.arange(
                20, num_comp), np.cumsum(
                pca.explained_variance_ratio_)[
                20:], 'bo')
        plt.title('Principle Component Analysis - MRI Segmentation')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.savefig(
            history_save_location +
            '/' +
            'pca_excl_early_range.png',
            format='png')
        np.save(history_save_location + '/' + 'pca_cumsum',
                np.cumsum(pca.explained_variance_ratio_))

        # np.save('intermediate_model_output_test_data',np.array(intermediate_prediction))
        print(intermediate_prediction.shape)

        print('Exiting after intermediate model prediction, back to normal fitting process')
        exit()

    model.compile(loss=dice_coef_loss_by_sample(),
                  optimizer=Adam(learning_rate=params['lr']),
                  metrics=[dice_coef, jaccard_similarity])
    print('Model compiled. Moving to loading weights')

    model.load_weights('start_models/initialWeights.h5',
                       by_name=False,
                       skip_mismatch=False)

    # Make the necessary ingredients for saving modality metrics on a
    # per-epoch basis
    modality_filename = 'modality_results_' + str(time.time()) + '.csv'
    modality_metric_save_location = history_save_location + '/' + modality_filename
    column_names = [
        'anatomical_val_dice',
        'dti_val_dice',
        'noddi_val_dice',
        'regwarp_val_dice',
        'fmri_val_dice',
        'anatomical_train_dice',
        'dti_train_dice',
        'noddi_train_dice',
        'regwarp_train_dice',
        'fmri_train_dice']
    modality_results = pd.DataFrame(columns=column_names)
    modality_results.to_csv(modality_metric_save_location)

    if params['modality_weight'] is None:
        print('Weights loaded, moving to model fit')
        history = model.fit(
            x_train_data,
            y_train_data,
            validation_data=(
                x_val_data,
                y_val_data),
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            verbose=1,
            callbacks=[
                modalityMetrics(
                    x_train_data,
                    y_train_data,
                    x_val_data,
                    y_val_data,
                    anatomical_val_indices,
                    dti_val_indices,
                    noddi_val_indices,
                    regwarp_val_indices,
                    fmri_val_indices,
                    anatomical_train_indices,
                    dti_train_indices,
                    noddi_train_indices,
                    regwarp_train_indices,
                    fmri_train_indices,
                    modality_metric_save_location)])
        #           talos.utils.early_stopper(monitor='val_loss',mode=[0,100])])
    if params['modality_weight'] is not None:
        print('Weights loaded, moving to model fit - including varying weight by modality')
        history = model.fit(x_train_data, y_train_data,
                            validation_data=(x_val_data, y_val_data),
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            verbose=1,
                            sample_weight=sample_weight)
        # callbacks=[modalityMetrics(x_train_data,
        #                           y_train_data,
        #                           x_val_data,
        #                           y_val_data,
        #                           anatomical_val_indices,
        #                           dti_val_indices,
        #                           noddi_val_indices,
        #                           regwarp_val_indices,
        #                           fmri_val_indices,
        #                           anatomical_train_indices,
        #                           dti_train_indices,
        #                           noddi_train_indices,
        #                           regwarp_train_indices,
        #                           fmri_train_indices,
        #                           modality_metric_save_location)])
        #           talos.utils.early_stopper(monitor='val_loss',mode=[0,100])])

    # Save the model history so we can look at what's happening inside each
    # talos run
    history_filename = 'talos_scan_history_' + str(time.time()) + '.csv'
    history_df = pd.DataFrame(history.history)
    # Save the file
    print('A Talos Loop Has Been Completed. Saving Model History as ' +
          history_save_location + '/' + history_filename)
    history_df.to_csv(history_save_location + '/' + history_filename)
    print(
        'Saving Modality Results as ' +
        history_save_location +
        '/' +
        modality_filename)

    return history, model
