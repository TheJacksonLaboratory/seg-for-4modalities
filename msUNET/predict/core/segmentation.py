from .corrections import y_axis_correction
from .corrections import z_axis_correction
from ..scripts.rbm import brain_seg_prediction
from .quality import quality_check
from ..scripts.original_seg import brain_seg_prediction_original
from .utils import get_suffix
import numpy as np
import SimpleITK as sitk
import glob
import os
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
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

def segment_brain(source_fn,
                  z_axis_correction_check,
                  y_axis_correction_check,
                  voxsize,
                  pre_paras,
                  keras_paras,
                  new_spacing,
                  normalization_mode,
                  constant_size,
                  use_frac_patch,
                  likelihood_categorization,
                  y_axis_mask,
                  frac_patch,
                  frac_stride,
                  quality_checks,
                  qc_skip_edges,
                  target_size):

    suffix = get_suffix(z_axis_correction_check, y_axis_correction_check)
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

    if z_axis_correction_check == 'True':
        # Run z-axis correction, producing modified data
        z_axis_fn = str(source_path_obj.with_name(source_path_obj.stem.split('.')[0] + '_z_axis' + ''.join(source_path_obj.suffixes)))
        z_axis_path_obj = PurePath(z_axis_fn)
        #z_axis_fn = Path(source_fn).stem.split('.')[0] + '_z_axis.nii'
        # print(z_axis_fn)
        print('Performing z-axis correction')
        if not use_frac_patch:
            if not constant_size:
                z_axis_correction(
                    source_fn,
                    z_axis_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    new_spacing,
                    normalization_mode,
                    likelihood_categorization=likelihood_categorization)
            elif constant_size:
                z_axis_correction(
                    source_fn,
                    z_axis_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    new_spacing,
                    normalization_mode,
                    target_size,
                    likelihood_categorization=likelihood_categorization)
        elif use_frac_patch:
            if not constant_size:
                z_axis_correction(
                    source_fn,
                    z_axis_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    new_spacing,
                    normalization_mode,
                    frac_patch=frac_patch,
                    frac_stride=frac_stride,
                    likelihood_categorization=likelihood_categorization)
            elif constant_size:
                z_axis_correction(
                    source_fn,
                    z_axis_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    new_spacing,
                    normalization_mode,
                    target_size,
                    frac_patch=frac_patch,
                    frac_stride=frac_stride,
                    likelihood_categorization=likelihood_categorization)
    if y_axis_correction_check == 'True':
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
            new_spacing,
            y_axis_mask)

        if z_axis_correction_check == 'True':
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
                new_spacing,
                y_axis_mask)

    # Do the final inference
    final_inference_fn = str(source_path_obj.with_name(source_path_obj.stem.split('.')[0] + suffix + ''.join(source_path_obj.suffixes)))
    #final_inference_fn = Path(source_fn).stem.split('.')[0] + suffix + '.nii'
    # print(final_inference_fn)
    mask_fn = str(source_path_obj.with_name(source_path_obj.stem.split('.')[0] + '_mask' + ''.join(source_path_obj.suffixes)))
    #mask_fn = Path(source_fn).stem.split('.')[0] + '_mask.nii'
    # print(final_inference_fn)
    if not use_frac_patch:
        if not constant_size:
            brain_seg_prediction(
                final_inference_fn,
                mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                likelihood_categorization=likelihood_categorization)
        elif constant_size:
            new_spacing = None
            brain_seg_prediction(
                final_inference_fn,
                mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                target_size,
                likelihood_categorization=likelihood_categorization)
    if use_frac_patch:
        if not constant_size:
            brain_seg_prediction(
                final_inference_fn,
                mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                frac_patch=frac_patch,
                frac_stride=frac_stride,
                likelihood_categorization=likelihood_categorization)
        elif constant_size:
            new_spacing = None
            brain_seg_prediction(
                final_inference_fn,
                mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                target_size,
                frac_patch=frac_patch,
                frac_stride=frac_stride,
                likelihood_categorization=likelihood_categorization)
    # If everything ran well up to this point, clean up the backup file and put the source .nii
    # back where it belongs
    shutil.copyfile(original_fn, source_fn)
    os.remove(original_fn)

    # Do some post-inference quality checks
    # Often overlap with each other and with low SNR. Can catch unique
    # cases though.
    quality_check_list = pd.DataFrame(columns=['filename', 'slice_index', 'notes'])
    if quality_checks:
        print('Performing post-inference quality checks')
        source_array = sitk.GetArrayFromImage(sitk.ReadImage(source_fn))
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_fn))
        qc_classifier = joblib.load('./msUNET/predict/scripts/quality_check_11822.joblib')
        file_quality_check_df = quality_check(source_array, mask_array, qc_classifier, source_fn, mask_fn, qc_skip_edges)
        quality_check_list = quality_check_list.append(file_quality_check_df, ignore_index=True)

    return quality_check_list