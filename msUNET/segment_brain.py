from predict.core.corrections import y_axis_correction
from predict.core.corrections import z_axis_correction
from predict.core.utils import str2bool
from predict.core.utils import solidity_check
from predict.core.utils import intermediate_likelihood_check
from predict.core.utils import mask_area_check
from predict.core.utils import low_snr_check
from predict.core.utils import get_suffix
from predict.core.utils import listdir_nohidden
from predict.scripts.rbm import brain_seg_prediction
from predict.core.quality import quality_check
from predict.scripts.original_seg import brain_seg_prediction_original
from pathlib import PurePath
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
    "-i",
    "--input",
    help="Directory containing specified file structure",
    type=str,
    required=False)
parser.add_argument(
    "-m",
    "--model",
    help="Filename of the model to be used in inference",
    type=str,
    default='model156_0.50.25_scan.hdf5')
parser.add_argument(
    "-sp",
    "--skip_preprocessing",
    help="Skip all preprocessing steps and use original segmentation algorithm",
    type=str2bool,
    default=False)
parser.add_argument(
    "-if",
    "--input_filename",
    help="Input filename of single file to run inference on",
    type=str,
    default=None)
parser.add_argument(
    "-th",
    "--threshold",
    help="Score threshold for selecting brain region",
    type=float,
    default=0.5)
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
    default=False)
parser.add_argument(
    "-st",
    "--low_snr_threshold",
    help="Multiplicative threshold below which low SNR warning flag will be thrown",
    type=float,
    default=3.5)
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
        280,
        280,
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
    default=0.25,
    required=False,
    type=float)
parser.add_argument(
    "-se",
    "--qc_skip_edges",
    default=False,
    type=str2bool)

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
keras_paras.model_path = './predict/scripts/' + opt.model

# Declare input variables that generally remain constant in the instance
# of constant input type
voxsize = 0.1

if opt.skip_preprocessing:
    print('Skipping all preprocessing steps...')
    if opt.input_filename is not None:
        input_path_obj = PurePath(opt.input_filename)
        output_filename  = str(input_path_obj.with_name(input_path_obj.stem.split('.')[0] + '_mask' + ''.join(input_path_obj.suffixes)))
        print(output_filename)
        brain_seg_prediction_original(opt.input_filename, output_filename, voxsize, pre_paras, keras_paras)
        exit()

quality_check_list = pd.DataFrame(columns=['filename', 'slice_index', 'notes'])

mouse_dirs = sorted(listdir_nohidden(opt.input))

print('Working with the following dataset directory: ' + opt.input)
print(
    'It contains the following subdirectories corresponding to individual mice: \n' +
    str(mouse_dirs))

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
        # print(source_fn)
        suffix = get_suffix(opt.z_axis_correction, opt.y_axis_correction)
        # print(suffix)

        # Write a copy of the source image to original_fn_original.nii
        source_path_obj = PurePath(source_fn)
        original_fn = str(source_path_obj.with_name(source_path_obj.stem.split('.')[0] + '_backup' + ''.join(source_path_obj.suffixes)))
        #original_fn = Path(source_fn).stem.split('.')[0] + '_backup.nii'
        shutil.copyfile(source_fn, original_fn)

        # Do some very basic data preparation
        source_img = sitk.ReadImage(source_fn)
        source_spacing = source_img.GetSpacing()
        # Check for images with an extra dimension (NODDI). If they have one,
        # use only the 8th frame
        dim_check_array = sitk.GetArrayFromImage(source_img)
        if len(dim_check_array.shape) > 3:
            inference_array = dim_check_array[7, :, :, :]
            inference_img = sitk.GetImageFromArray(inference_array)
            inference_img.SetSpacing(source_spacing)
            sitk.WriteImage(inference_img, source_fn)
        # Clip data points that are far above the mean
        source_image = sitk.ReadImage(source_fn)
        source_array = sitk.GetArrayFromImage(source_image)
        source_shape = source_array.shape
        clip_value = np.mean(source_array) * 20
        replace_value = np.median(source_array)
        source_array = np.where(
            source_array > clip_value,
            replace_value,
            source_array)
        source_array = np.reshape(source_array, source_shape)
        source_image = sitk.GetImageFromArray(source_array)
        source_image.SetSpacing(source_spacing)
        #source_image = sitk.Cast(source_image, sitk.sitk.UInt8)
        sitk.WriteImage(source_image, source_fn)

        if opt.z_axis_correction == 'True':
            # Run z-axis correction, producing modified data
            z_axis_fn = str(source_path_obj.with_name(source_path_obj.stem.split('.')[0] + '_z_axis' + ''.join(source_path_obj.suffixes)))
            z_axis_path_obj = PurePath(z_axis_fn)
            #z_axis_fn = Path(source_fn).stem.split('.')[0] + '_z_axis.nii'
            # print(z_axis_fn)
            print('Performing z-axis correction')
            if not opt.use_frac_patch:
                if not opt.constant_size:
                    z_axis_correction(
                        source_fn,
                        z_axis_fn,
                        voxsize,
                        pre_paras,
                        keras_paras,
                        opt.new_spacing,
                        opt.normalization_mode)
                elif opt.constant_size:
                    z_axis_correction(
                        source_fn,
                        z_axis_fn,
                        voxsize,
                        pre_paras,
                        keras_paras,
                        opt.new_spacing,
                        opt.normalization_mode,
                        opt.target_size)
            elif opt.use_frac_patch:
                if not opt.constant_size:
                    z_axis_correction(
                        source_fn,
                        z_axis_fn,
                        voxsize,
                        pre_paras,
                        keras_paras,
                        opt.new_spacing,
                        opt.normalization_mode,
                        frac_patch=opt.frac_patch,
                        frac_stride=opt.frac_stride)
                elif opt.constant_size:
                    z_axis_correction(
                        source_fn,
                        z_axis_fn,
                        voxsize,
                        pre_paras,
                        keras_paras,
                        opt.new_spacing,
                        opt.normalization_mode,
                        opt.target_size,
                        frac_patch=opt.frac_patch,
                        frac_stride=opt.frac_stride)
        if opt.y_axis_correction == 'True':
            # Run y-axis correction, producing modified data
            print('Performing y-axis correction to source data')
            y_axis_fn = str(source_path_obj.with_name(source_path_obj.stem.split('.')[0] + '_n4b' + ''.join(source_path_obj.suffixes)))
            #y_axis_fn = Path(source_fn).stem.split('.')[0] + '_n4b.nii'
            # print(y_axis_fn)
            y_axis_correction(
                source_fn,
                y_axis_fn,
                voxsize,
                pre_paras,
                keras_paras,
                opt.new_spacing,
                opt.y_axis_mask)

            if opt.z_axis_correction == 'True':
                print('Performing y-axis correction to z-axis corrected data')
                # If we have already done a z-axis correction, do a y axis correction on that file too.
                # The file created with n4b alone is simply intended to be a
                # check
                z_axis_n4b_fn = str(z_axis_path_obj.with_name(z_axis_path_obj.stem.split('.')[0] + '_n4b' + ''.join(z_axis_path_obj.suffixes)))
                #z_axis_n4b_fn = Path(z_axis_fn).stem.split('.')[0] + '_n4b.nii'
                # print(z_axis_n4b_fn)
                y_axis_correction(
                    z_axis_fn,
                    z_axis_n4b_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    opt.new_spacing,
                    opt.y_axis_mask)

        # Do the final inference
        final_inference_fn = str(source_path_obj.with_name(source_path_obj.stem.split('.')[0] + suffix + ''.join(source_path_obj.suffixes)))
        #final_inference_fn = Path(source_fn).stem.split('.')[0] + suffix + '.nii'
        # print(final_inference_fn)
        mask_fn = str(source_path_obj.with_name(source_path_obj.stem.split('.')[0] + '_mask' + ''.join(source_path_obj.suffixes)))
        #mask_fn = Path(source_fn).stem.split('.')[0] + '_mask.nii'
        # print(final_inference_fn)
        if not opt.use_frac_patch:
            if not opt.constant_size:
                brain_seg_prediction(
                    final_inference_fn,
                    mask_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    opt.new_spacing,
                    opt.normalization_mode)
            elif opt.constant_size:
                opt.new_spacing = None
                brain_seg_prediction(
                    final_inference_fn,
                    mask_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    opt.new_spacing,
                    opt.normalization_mode,
                    opt.target_size)
        if opt.use_frac_patch:
            if not opt.constant_size:
                brain_seg_prediction(
                    final_inference_fn,
                    mask_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    opt.new_spacing,
                    opt.normalization_mode,
                    frac_patch=opt.frac_patch,
                    frac_stride=opt.frac_stride)
            elif opt.constant_size:
                opt.new_spacing = None
                brain_seg_prediction(
                    final_inference_fn,
                    mask_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    opt.new_spacing,
                    opt.normalization_mode,
                    opt.target_size,
                    frac_patch=opt.frac_patch,
                    frac_stride=opt.frac_stride)
        # If everything ran well up to this point, clean up the backup file and put the source .nii
        # back where it belongs
        shutil.copyfile(original_fn, source_fn)
        os.remove(original_fn)

        # Do some post-inference quality checks
        # Often overlap with each other and with low SNR. Can catch unique
        # cases though.
        if opt.quality_checks:
            print('Performing post-inference quality checks')
            source_array = sitk.GetArrayFromImage(sitk.ReadImage(source_fn))
            mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_fn))
            qc_classifier = joblib.load('./predict/scripts/quality_check_11822.joblib')
            file_quality_check_df = quality_check(source_array, mask_array, qc_classifier, source_fn, mask_fn, opt.qc_skip_edges)
            quality_check_list = quality_check_list.append(file_quality_check_df, ignore_index=True)

print(quality_check_list)
if len(quality_check_list) > 0:
    print('Saving quality check file to: ' + opt.input + 'quality_check.csv')
    quality_check_list.to_csv(opt.input + '/quality_check.csv', index=False)

