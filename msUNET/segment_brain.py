from predict.core.corrections import y_axis_correction
from predict.core.corrections import z_axis_correction
from predict.core.utils import str2bool
from predict.core.utils import solidity_check
from predict.core.utils import intermediate_likelihood_check
from predict.core.utils import mask_area_check
from predict.core.utils import low_snr_check
from predict.core.utils import listdir_nohidden
from predict.core.segmentation import segment_brain
from predict.scripts.rbm import brain_seg_prediction
from predict.core.quality import quality_check
from predict.scripts.original_seg import brain_seg_prediction_original
from pathlib import PurePath
from contextlib import redirect_stdout
import joblib
import shutil
import SimpleITK as sitk
import argparse
import glob
import sys
import pprint
import numpy as np
import pandas as pd
import os
import warnings
import time

sitk.ProcessObject_SetGlobalWarningDisplay(False)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Main function that handles inference for segmentation tasks.
# See User Guide for detailed information about inputs and outputs!
# INPUTS:
# Required:
# input: Full path to a directory structured in the following way:
# - $Dataset-Name -> - $mouseName1 -> - $modality1 -> $mouseName1_modality1.nii
#									  - $modality2 -> $mouseName1_modality2.nii
#									  - $modality3 -> $mouseName1_modality3.nii
#				     - $mouseName2 -> - $modality1 -> $mouseName1_modality1.nii
#									  - $modality2 ... and so on
# Optional:
# model: The model you'd like to use to do inference.
# 		 Currently assumes that the models live in ./predict/scripts
# threshold: The score threshold at which the inference declares a pixel to be brain
# channel_location: Whether image channels are the first or last dimension. Channels implying image slices
# z_axis_correction: Whether to normalize intensity across image slices in a .nii file.
# 					 Requires running inference twice. In the first we find a preliminary mask used to take
#					 the are used to calculate average intensity. Intensity is adjusted, then inferred again.
#					 Will roughly double runtime.
# y_axis_correction: Whether to normalize intensity across the y-axis of single slices.
# new_spacing:
# OUTPUTS:
# Required:
# $mouseName_$modality_$mask.nii: Mask containing segmentation results for the given mouse/modality combination.
#							Will be located in the same folder as the raw data source
# Optional:
# The following only exist if the corresponding corrections have been applied. If they are applied, the corresponding
# file with the *most* corrections will be used as a source for inference.
# $mouseName_$modality_$z_axis.nii: Data with z-axis correction applied. Will again be saved in the source folder
# $mouseName_$modality_$n4b.nii: Data with n4bias (y-axis) corrections applied. Saved in source folder
# $mouseName_$modality_$z_axis_$n4b.nii: Data with both n4bias and z-axis corrections applied. Same save location
parser = argparse.ArgumentParser()
parser.add_argument(
    "-it",
    "--input_type",
    type=str,
    required=True,
    choices=['dataset',
             'directory',
             'file'])
parser.add_argument(
    "-i",
    "--input",
    help="Directory containing specified file structure",
    type=str,
    required=True)
parser.add_argument(
    "-m",
    "--model",
    help="Filename of the model to be used in inference",
    type=str,
    default='model156_0.50.25_scan.hdf5')
parser.add_argument(
    "-th",
    "--threshold",
    help="Score threshold for selecting brain region",
    type=float,
    default=0.3)
parser.add_argument(
    "-cl",
    "--channel_location",
    help="Whether the channels are the first or last dimension",
    type=str,
    default='channels_last',
    choices=[
        'channels_first',
         'channels_last'])
parser.add_argument(
    "-zc",
    "--z_axis_correction",
    help="Whether to perform z axis correction",
    type=str,
    default='False',
    choices=[
        'True',
         'False'])
parser.add_argument(
    "-yc",
    "--y_axis_correction",
    help="Whether to perform y axis correction",
    type=str,
    default='False',
    choices=[
        'True',
         'False'])
parser.add_argument(
    "-ns",
    "--new_spacing",
    help="Multiplicative factor by which image dimensions are divided. \
					For 3D images, 3 values. First two correspond to single-slice dimensions, third to between-slice dimension.",
    nargs="*",
    type=float,
    default=[
        0.08,
        0.08,
         1])  # usually 0.08,0.08,0.76
parser.add_argument(
    "-ip",
    "--image_patch",
    help="Dimensions of the image patches to use to perform inference. Best to use the patch size from model training",
    type=int,
    default=128)
parser.add_argument(
    "-is",
    "--image_stride",
    help="Stride of the image patching method to use in inference. Best to use the stride corresponding to that used in training",
    type=int,
    default=16)
parser.add_argument(
    "-qc",
    "--quality_checks",
    help="Perform pre/post inference quality checks. If true, will output a list of IDs + slices that have an issue",
    type=str2bool,
    default=True)
parser.add_argument(
    "-se",
    "--qc_skip_edges",
    default=False,
    type=str2bool)
parser.add_argument(
    "-nt",
    "--normalization_mode",
    help="Use either by_slice or by_image to normalize images pre-inference",
    type=str,
    default="by_img",
    choices=[
        'by_slice',
         'by_img'])
parser.add_argument(
    "-ym",
    "--y_axis_mask",
    help="To use otsu threshold to mask y-axis correction pixels",
    type=str2bool,
    default=True)
parser.add_argument(
    "-cs",
    "--constant_size",
    default=False,
    required=False,
    type=str2bool)
parser.add_argument(
    "-ts",
    "--target_size",
    default=[
        188,
        188,
        None],
    nargs="*",
    type=int,
    required=False)
parser.add_argument(
    "-ufp",
    "--use_frac_patch",
    default=False,
    required=False,
    type=str2bool)
parser.add_argument(
    "-fp",
    "--frac_patch",
    default=0.5,
    required=False,
    type=float)
parser.add_argument(
    "-fs",
    "--frac_stride",
    default=0.125,
    required=False,
    type=float)
parser.add_argument(
    "-lc",
    "--likelihood_categorization",
    default=True,
    type=str2bool)
parser.add_argument( # Deprecated debug option
    "-sp",
    "--skip_preprocessing",
    help="Skip all preprocessing steps and use original segmentation algorithm",
    type=str2bool,
    default=False)

opt = parser.parse_args()

# Initialize parameter classes


class KerasParas:
    def __init__(self):
        self.model_path = None
        self.outID = 0
        self.thd = 0.5
        self.img_format = 'channels_last'
        self.loss = None


class PreParas:
    def __init__(self):
        self.patch_dims = []
        self.patch_label_dims = []
        self.patch_strides = []
        self.n_class = ''

# Prep parameter classes with input information or, failing that, defaults


# Parameters for image processing
pre_paras = PreParas()
# Dimensions and stride of image patches on which the CNN was trained. Can
# be different, but performance decreases
pre_paras.patch_dims = [1, opt.image_patch, opt.image_patch]
pre_paras.patch_label_dims = [1, opt.image_patch, opt.image_patch]
pre_paras.patch_strides = [1, opt.image_stride, opt.image_stride]
pre_paras.n_class = 2

# Parameters for Keras model
keras_paras = KerasParas()
keras_paras.outID = 0
keras_paras.thd = opt.threshold
keras_paras.loss = 'dice_coef_loss'
keras_paras.img_format = opt.channel_location
keras_paras.model_path = './msUNET/predict/scripts/' + opt.model

# Declare input variables that generally remain constant in the instance
# of constant input type
voxsize = 0.1

quality_check = pd.DataFrame(columns=['filename', 'slice_index', 'notes'])

if opt.input_type == 'dataset':
    mouse_dirs = sorted(listdir_nohidden(opt.input))
    print('Working with the following dataset directory: ' + opt.input)
    print(
        'It contains the following subdirectories corresponding to individual mice: \n' +
        str(mouse_dirs))
    input_path_obj = PurePath(opt.input)
    qc_log_path = str(opt.input + '/segmentation_log.txt')
    sys.stderr = open(qc_log_path, 'w')
    for mouse_dir in mouse_dirs:
        # Determine what modalities will be evaluated for each mouse.
        modality_dirs = sorted(listdir_nohidden(mouse_dir))
        print(
            'For the mouse ' +
            str(mouse_dir) +
            ' I see the following modality folders: \n ' +
            str(modality_dirs))
        for modality_dir in modality_dirs:
            # For each mouse/modality combination we will run the inference
            source_fn = glob.glob(os.path.join(modality_dir, '*'))[0]
            print('Starting Inference on file: ' + source_fn)
            quality_check_temp = segment_brain(source_fn,
                                               opt.z_axis_correction,
                                               opt.y_axis_correction,
                                               voxsize,
                                               pre_paras,
                                               keras_paras,
                                               opt.new_spacing,
                                               opt.normalization_mode,
                                               opt.constant_size,
                                               opt.use_frac_patch,
                                               opt.likelihood_categorization,
                                               opt.y_axis_mask,
                                               opt.frac_patch,
                                               opt.frac_stride,
                                               opt.quality_checks,
                                               opt.qc_skip_edges,
                                               opt.target_size)
            quality_check = quality_check.append(quality_check_temp, ignore_index=True)
    sys.stderr.close()
    sys.stderr = sys.__stderr__

elif opt.input_type == 'directory':
    print('Working with the following directory: ' + opt.input)
    print('It contains the following data files: \n' + 
        str(listdir_nohidden(opt.input)))
    source_files = listdir_nohidden(opt.input)
    input_path_obj = PurePath(opt.input)
    qc_log_path = str(opt.input + '/segmentation_log.txt')
    sys.stderr = open(qc_log_path, 'w')
    for source_fn in source_files:
        print('Starting Inference on file: ' + source_fn)
        quality_check_temp = segment_brain(source_fn,
                                           opt.z_axis_correction,
                                           opt.y_axis_correction,
                                           voxsize,
                                           pre_paras,
                                           keras_paras,
                                           opt.new_spacing,
                                           opt.normalization_mode,
                                           opt.constant_size,
                                           opt.use_frac_patch,
                                           opt.likelihood_categorization,
                                           opt.y_axis_mask,
                                           opt.frac_patch,
                                           opt.frac_stride,
                                           opt.quality_checks,
                                           opt.qc_skip_edges,
                                           opt.target_size)
        quality_check = quality_check.append(quality_check_temp, ignore_index=True)
    sys.stderr.close()
    sys.stderr = sys.__stderr__

elif opt.input_type == 'file':
    if opt.skip_preprocessing == True: # Debug option, currently disabled
        print('Skipping all preprocessing steps...')
        if opt.input is not None:
            input_path_obj = PurePath(opt.input)
            output_filename  = str(input_path_obj.with_name(input_path_obj.stem.split('.')[0] + '_mask' + ''.join(input_path_obj.suffixes)))
            print(output_filename)
            brain_seg_prediction_original(opt.input, output_filename, voxsize, pre_paras, keras_paras, opt.likelihood_categorization)
            exit()
    print('Performing inference on the following file: ' + str(opt.input))
    source_fn = opt.input
    print('Starting Inference on file: ' + source_fn)
    input_path_obj = PurePath(opt.input)
    qc_log_path = str(input_path_obj.parents[0]) + '/segmentation_log.txt'
    sys.stderr = open(qc_log_path, 'w')
    quality_check_temp = segment_brain(source_fn,
                                   opt.z_axis_correction,
                                   opt.y_axis_correction,
                                   voxsize,
                                   pre_paras,
                                   keras_paras,
                                   opt.new_spacing,
                                   opt.normalization_mode,
                                   opt.constant_size,
                                   opt.use_frac_patch,
                                   opt.likelihood_categorization,
                                   opt.y_axis_mask,
                                   opt.frac_patch,
                                   opt.frac_stride,
                                   opt.quality_checks,
                                   opt.qc_skip_edges,
                                   opt.target_size)
    sys.stderr.close()
    sys.stderr = sys.__stderr__
    quality_check = quality_check.append(quality_check_temp, ignore_index=True)

if len(quality_check) > 0:
    if opt.input_type == 'file':
        print('Saving quality check file to: ' + str(input_path_obj.parents[0]) + '/quality_check.csv')
        quality_check.to_csv(str(input_path_obj.parents[0]) + '/quality_check.csv', index=False)        
    else:   
        print('Saving quality check file to: ' + opt.input + 'quality_check.csv')
        quality_check.to_csv(opt.input + '/quality_check.csv', index=False)


