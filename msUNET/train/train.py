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
from metrics import dice_coef, dice_coef_metric, dice_coef_loss, jaccard_similarity, jaccard_distance
from util import layer_parse, save_object, remove_x_y_data, load_data
from augmentation import rotation_augmentation, affine_augmentation, full_augmentation
from model import mouseSegUNET

# Import some various useful packages
import os
import argparse
import math
import time
import shutil
from zipfile import ZipFile
import pickle
import gc
import random


def train_seg_net(initial_model_path,
                  # Function that trains the segmentation neural network.
                  img_params,
                  training_data_path,
                  training_mask_path,
                  interpolation_spacing,
                  talos_params,
                  validation_split,
                  experiment_name,
                  enable_rotation,
                  enable_affine,
                  enable_noise,
                  enable_contrast_change,
                  save_directory,
                  constant_size,
                  target_size,
                  use_frac_patch,
                  frac_patch,
                  frac_stride,
                  in_place_augmentation,
                  augmentation_threshold,
                  modality_weight,
                  weight_factor):

    print('CONSTANT SIZE: ' + str(constant_size))
    # Load weights from initial weights path
    initial_model = load_model(
        initial_model_path,
        custom_objects={
            'dice_coef_loss': dice_coef_loss,
            'dice_coef': dice_coef})
    # initial_model.save_weights('initial_weights.h5')

    # Gather list of training data and mask names
    training_data_list = sorted(os.listdir(training_data_path))
    mask_data_list = sorted(os.listdir(training_mask_path))

    # Make mask format uniform
    uniform_mask(training_mask_path, mask_data_list)

    if not in_place_augmentation:

        print('Starting Data Augmentation - Rotation')
        # Do data augmentation if requested
        if enable_rotation:
            rotation_augmentation(
                training_data_path,
                training_mask_path,
                training_data_list,
                mask_data_list)
        print('Finished Data Augmentation - Rotation')

        print("Starting Data Augmentation - Affine")
        # Reload training and mask data paths to reflect augmented data
        training_data_list = sorted(os.listdir(training_data_path))
        mask_data_list = sorted(os.listdir(training_mask_path))

        if enable_affine:
            affine_augmentation(
                training_data_path,
                training_mask_path,
                training_data_list,
                mask_data_list)
        print('Finished Data Augmentation - Affine')

    if in_place_augmentation:

        full_augmentation(
            training_data_path,
            training_mask_path,
            training_data_list,
            mask_data_list,
            enable_rotation,
            enable_affine,
            enable_noise,
            enable_contrast_change,
            augmentation_threshold)

    # Reload training and mask data paths to reflect augmented data
    training_data_list = sorted(os.listdir(training_data_path))
    mask_data_list = sorted(os.listdir(training_mask_path))

    # Build empty list of training data sets
    training_data = []
    mask_data = []
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    scan_list = []
    iter_count = 0
    modality_list = []
    # Create all training and validation data sets
    for patch_dimension in list(talos_params['patch_dimensions']):
        for patch_stride in list(talos_params['patch_stride']):
            # Create image parameters
            img_params.patch_dims = [1, patch_dimension, patch_dimension]
            img_params.patch_strides = [1, patch_stride, patch_stride]

            # Determine what the name of the dataset should be. Of the form *patch_dim*_*patch_stride*-*source_dataset_name*
            ##original_dataset_name = training_data_path.split('/')[0]
            ##current_expanded_dataset_name = str(img_params.patch_dims[1])+'_'+str(img_params.patch_strides[1])+'-'+original_dataset_name

            # Check if the dataset directory we're looking at exists and has some files
            # If it does, do not write anything, move to the next
            # if os.path.isdir(save_directory+'/'+current_expanded_dataset_name) and len(os.listdir(save_directory+'/'+current_expanded_dataset_name+'/training')) != 0:
            ##    print('Dataset found for patch size '+str(img_params.patch_dims[1])+' stride '+str(img_params.patch_strides[1])+' for dataset '+current_expanded_dataset_name)
            ##    print('No need to generate, moving to the next')
            # continue

            ##print('Generating Dataset for patch size '+str(img_params.patch_dims[1])+' stride '+str(img_params.patch_strides[1])+' for dataset '+current_expanded_dataset_name)

            # Check the number of training and mask files. If they are not the
            # same, error
            num_files = len(training_data_list)
            num_masks = len(mask_data_list)
            if num_masks != num_files:
                raise ValueError(
                    'Mismatch in number of training images and masks')

            ##expanded_dataset_dir_path = save_directory+'/'+current_expanded_dataset_name

            # Loop through the files in the source dataset, saving each image patch as an individual .npy file
            # We must do this for each image patch and stride, as it serves as what amounts to a new dataset each time
            # for i in range(0,num_files):
            ##    current_modality_list = [4]
                #current_modality_list = load_data(expanded_dataset_dir_path, training_data_path, training_data_list[i], training_mask_path, mask_data_list[i], interpolation_spacing, img_params)
            # modality_list.append(current_modality_list)

            # Save the modality list in the new expanded dataset directory. Though it's not likley to be necessary,
                # it could help decrease execution time later
            # print(current_expanded_dataset_name)
            # with open(save_directory+'/'+current_expanded_dataset_name+'/modality_list.npy','wb') as file:
            # np.save(file,modality_list)

           # exit()

            print("Loading Data...")
            # Load training data and masks
            #training_data_temp = load_data(training_data_path, training_data_list, interpolation_spacing, img_params, False)
            #mask_data_temp = load_data(training_mask_path, mask_data_list, interpolation_spacing, img_params, True)
            if not use_frac_patch:
                print('Creating patches based on provided fixed patch size and stride')
                if not constant_size:
                    print('Resizing images based on fixed multiplicative factor')
                    training_data_temp, mask_data_temp, modality_list_temp = load_data(
                        training_data_path, training_data_list, training_mask_path, mask_data_list, interpolation_spacing, img_params)
                elif constant_size:
                    print(
                        'Resizing images to consistent size with aspect ratio from original')
                    training_data_temp, mask_data_temp, modality_list_temp = load_data(
                        training_data_path, training_data_list, training_mask_path, mask_data_list, interpolation_spacing, img_params, target_size)
            elif use_frac_patch:
                print('Creating dynamically sized patches equal to a fraction of image dimensions - Resulting images padded to behave nicely')
                if not constant_size:
                    print('Resizing images based on fixed multiplicative factor')
                    training_data_temp, mask_data_temp, modality_list_temp, patch_dim1, patch_dim2 = load_data(
                        training_data_path, training_data_list, training_mask_path, mask_data_list, interpolation_spacing, img_params, frac_patch=frac_patch, frac_stride=frac_stride)
                elif constant_size:
                    print(
                        'Resizing images to consistent size with aspect ratio from original')
                    training_data_temp, mask_data_temp, modality_list_temp, patch_dim1, patch_dim2 = load_data(
                        training_data_path, training_data_list, training_mask_path, mask_data_list, interpolation_spacing, img_params, target_size, frac_patch=frac_patch, frac_stride=frac_stride)
                #img_params.patch_dims=[1, patch_dim1, patch_dim2]
                img_params.patch_dims = [1, patch_dim1, patch_dim2]
                img_params.patch_strides = [
                    1,
                    patch_dim1 /
                    frac_patch *
                    frac_stride,
                    patch_dim2 /
                    frac_patch *
                    frac_stride]
                patch_dimension = img_params.patch_dims[1]
                talos_params['patch_dimensions'] = [
                    str(patch_dim1) + '/' + str(patch_dim2)]
            if mask_data_temp.shape != training_data_temp.shape:
                raise ValueError(
                    'Number of training data and masks do not match.')
            # training_data.append(training_data_temp)
            # mask_data.append(mask_data_temp)

            talos_params['modality_weight'] = [modality_weight]
            talos_params['weight_factor'] = [weight_factor]

            #training_data_image = sitk.GetImageFromArray(training_data_temp)
            #mask_data_image = sitk.GetImageFromArray(mask_data_temp)
            #sitk.WriteImage(training_data_image, str(patch_dimension)+str(patch_stride)+'training_data.nii')
            #sitk.WriteImage(mask_data_image, str(patch_dimension)+str(patch_stride)+'mask_data.nii')
            # sitk.WriteImage(sitk.GetImageFromArray(training_data_temp),'training_test.nii')
            # sitk.WriteImage(sitk.GetImageFromArray(mask_data_temp),'mask_test.nii')

            # exit()

            # Split data into training and validation samples
            num_samples = training_data_temp.shape[0]
            x_train.append(str(patch_dimension) + '/' + str(patch_stride))
            x_train.append(modality_list_temp[0:math.floor(
                num_samples * (1 - validation_split))])
            x_train.append(training_data_temp[0:math.floor(
                num_samples * (1 - validation_split)), :, :, :])
            y_train.append(mask_data_temp[0:math.floor(
                num_samples * (1 - validation_split)), :, :, :])
            x_val.append(modality_list_temp[math.floor(
                num_samples * (1 - validation_split)) + 1:])
            x_val.append(training_data_temp[math.floor(
                num_samples * (1 - validation_split)) + 1:, :, :, :])
            y_val.append(mask_data_temp[math.floor(
                num_samples * (1 - validation_split)) + 1:, :, :, :])

            # We need to train on only 50 samples from the training set, and use the same validation set as before. Do so.
            #num_samples = training_data_temp.shape[0]
            # We must still include the patching dimension labeles
            # x_train.append(str(patch_dimension)+'/'+str(patch_stride))
            # Just take the first 55% of the data instead of the first 80% as training samples
            # x_train.append(modality_list_temp[0:math.floor(num_samples*(1-0.50))])
            # x_train.append(training_data_temp[0:math.floor(num_samples*(1-0.50)),:,:,:])
            # y_train.append(mask_data_temp[0:math.floor(num_samples*(1-0.50)),:,:,:])
            # The validation data must be identical
            # x_val.append(modality_list_temp[math.floor(num_samples*(1-validation_split))+1:])
            # x_val.append(training_data_temp[math.floor(num_samples*(1-validation_split))+1:,:,:,:])
            # y_val.append(mask_data_temp[math.floor(num_samples*(1-validation_split))+1:,:,:,:])

            iter_count += 1

    # Build some dummy data to hand to talos.Scan. It requires inputs for x and y, but when using dataset generators, those inputs
        # are not used.
    x_dummy = [[random.random() for i in range(5)] for j in range(5)]
    y_dummy = x_dummy

    # Run a Talos scan

    scan_object = talos.Scan(x=x_train,
                             y=y_train,
                             x_val=x_val,
                             y_val=y_val,
                             params=talos_params,
                             model=mouseSegUNET,
                             experiment_name=experiment_name)

    return scan_object


def main():

    # Set some default image parameters
    class default_img_params:
        patch_dims = [1, 128, 128]
        patch_label_dims = [1, 128, 128]
        patch_strides = [1, 32, 32]
        n_class = 2

    # Set default talos parameters that are not edited by command line
    # arguments
    default_talos_params = {
        'epochs': [20],
        'losses': [dice_coef_loss],
        'optimizer': [Adam],
        'lr': [.001],
        'batch_size': [128],
        'finalActivation': ['sigmoid'],
        'dropout': [0.5],
        'which_layer': [19],
        'patch_dimensions': [128],
        'patch_stride': [32],
        'scoreTrainable': [True],
        'expandingBatchNormTrainable': [True],
        'contractingBatchNormTrainable': [True],
        'upsamplingTrainable': [True],
        'upsamplingBatchNormTrainable': [True],
        'pca_only': [False]
    }
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tl",
        "--lowest_trainable_layer",
        default=19,
        type=float,
        required=False)
    parser.add_argument(
        "-path",
        "--initial_model_path",
        default='start_models/rat_brain-2d_unet.hdf5',
        required=False)
    parser.add_argument(
        "-img",
        "--img_params",
        default=default_img_params,
        required=False)
    parser.add_argument(
        "-tpath",
        "--training_data_path",
        default='dataset_fulltrain_noddival/training',
        required=False)
    parser.add_argument(
        "-mpath",
        "--training_mask_path",
        default='dataset_fulltrain_noddival/masks',
        required=False)
    parser.add_argument(
        "-hspace",
        "--horizontal_interpolation_spacing",
        default=0.08,
        type=float,
        required=False)
    parser.add_argument(
        "-vspace",
        "--interlayer_interpolation_spacing",
        default=0.76,
        type=float,
        required=False)
    parser.add_argument(
        "-talos",
        "--talos_params",
        default=default_talos_params,
        required=False)
    parser.add_argument(
        "-v",
        "--validation_split",
        default=0.0418,
        type=float,
        required=False)
    parser.add_argument("-n",
                        "--experiment_name",
                        default='experiment' + str(time.time()),
                        required=False)
    parser.add_argument(
        "-er",
        "--enable_rotation",
        default=True,
        required=False)
    parser.add_argument("-ea", "--enable_affine", default=True, required=False)
    parser.add_argument("-en", "--enable_noise", default=True, required=False)
    parser.add_argument(
        "-ec",
        "--enable_contrast_change",
        default=True,
        required=False)
    parser.add_argument(
        "-pd",
        "--patch_dimensions",
        default=[128],
        required=False)
    parser.add_argument("-ps", "--patch__stride", default=[32], required=False)
    parser.add_argument(
        "-sd",
        "--save_directory",
        default='/Users/frohoz/Documents/expanded_datasets',
        required=False)
    parser.add_argument(
        "-cs",
        "--constant_size",
        default=False,
        required=False)
    parser.add_argument(
        "-ts",
        "--target_size",
        default=[
            64,
            64,
            None],
        required=False)
    parser.add_argument(
        "-ufp",
        "--use_frac_patch",
        default=True,
        required=False)
    parser.add_argument("-fp", "--frac_patch", default=0.75, required=False)
    parser.add_argument("-fs", "--frac_stride", default=0.25, required=False)
    parser.add_argument(
        "-ia",
        "--in_place_augmentation",
        default=True,
        required=False)
    parser.add_argument(
        "-at",
        "--augmentation_threshold",
        default=0.5,
        required=False)
    parser.add_argument(
        "-mw",
        "--modality_weight",
        default='noddi',
        required=False)
    parser.add_argument("-wf", "--weight_factor", default=15, required=False)
    args = parser.parse_args()
    print(args)
    # Determine how many gpu are available for parallel processing purposes
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))

    # Build Talos Parameters that are altered by command line
    # default_talos_params['which_layer']=[args.lowest_trainable_layer]
    default_talos_params['experiment_name'] = [args.experiment_name]
    default_talos_params['num_gpus'] = [num_gpus]
    # default_talos_params['patch_dimensons']=[args.patch_dimensions]
    # default_talos_params['patch_stride']=[args.patch_stride]

    # tells the algorithm that you'd like each pixel to represent some distance in physical space
    # in our case, we have 17 vertical slices, and it works out that no interpolation requires a spacing of .76
    # in the horizontal direction, we have 13.12mmx15.75mm. A .1mm spacing
    # leads to an image size of 158x132
    interpolation_spacing = [
        args.horizontal_interpolation_spacing,
        args.horizontal_interpolation_spacing,
        args.interlayer_interpolation_spacing]

    # Build the talos scan object
    scan_obj = train_seg_net(args.initial_model_path,
                             args.img_params,
                             args.training_data_path,
                             args.training_mask_path,
                             interpolation_spacing,
                             args.talos_params,
                             args.validation_split,
                             args.experiment_name,
                             args.enable_rotation,
                             args.enable_affine,
                             args.enable_noise,
                             args.enable_contrast_change,
                             args.save_directory,
                             args.constant_size,
                             args.target_size,
                             args.use_frac_patch,
                             args.frac_patch,
                             args.frac_stride,
                             args.in_place_augmentation,
                             float(args.augmentation_threshold),
                             args.modality_weight,
                             args.weight_factor)

    # Deploy the talos model to a folder for comparison with other models from
    # batch scan
    model_name_save = 'model_deploy' + str(args.experiment_name)
    talos.Deploy(
        scan_object=scan_obj,
        model_name=model_name_save,
        metric='dice_coef')

    # Since talos deploy will not be able to restore data with more than 2 dimensions, we have to write some dummy
    # data ourselves
    zip_location = str(model_name_save) + ".zip"
    zip_name = zip_location.replace(".zip", '')
    fixed_zip_name = zip_name + 'fixed' + '.zip'
    remove_x_y_data(zip_location, 'temp.zip', '_x.csv')
    remove_x_y_data('temp.zip', zip_name + 'fixed' + '.zip', '_y.csv')

    # Make the dummy data
    data = pd.DataFrame(
        np.random.randint(
            0, 100, size=(
                2, 2)), columns=list('AB'))
    data.to_csv(zip_name + '_x.csv', header=None, index=None)
    data.to_csv(zip_name + '_y.csv', header=None, index=None)
    # Write the new data
    with ZipFile(fixed_zip_name, 'a') as z:
        z.write(zip_name + '_x.csv')
        z.write(zip_name + '_y.csv')
    # clean up leftover files
    os.remove('temp.zip')
    os.remove(zip_name + '_x.csv')
    os.remove(zip_name + '_y.csv')
    os.remove(zip_location)
    shutil.move(str(model_name_save) + "fixed.zip",
                str(args.experiment_name) + "/" + str(model_name_save) + ".zip")


if __name__ == '__main__':
    main()
