# author: Zachary Frohock
'''
Functions handling pre-inference image processing
'''

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from ..scripts.segmentation import brain_seg_prediction
from .utils import erode_img_by_slice, plot_intensity_comparison


def z_axis_correction(
        input_fn,
        output_fn,
        voxsize,
        pre_paras,
        keras_paras,
        new_spacing,
        normalization_mode,
        target_size=None,
        frac_patch=None,
        frac_stride=None,
        likelihood_categorization=True):
    '''
    Function that handles performing z-axis correction
    Z-axis correction attempts to normalize the intensity of brain regions
    across slices. To do so, it calculates a preliminary mask on raw data.
    That preliminary mask is eroded to increase the likelihood that brain
    tissue is all that is contained within. The raw image intensity in that
    region is then calculated,and used to calculate a factor by which each
    slice should be multiplied to normalize brain region intensity.
    INPUTS:
    input_fn: full file path of raw data source
    output_fn: full file path of output z-axis corrected data
    voxsize: size to which 3d voxels should be resampled.
             Generally kept at 0.1 for consistency
    pre_paras: Set of parameters to do with image patching
    keras_paras: Set of parameters to do with inference,
                 most relevant is classification threshold thd
    new_spacing: Muliplicative factor by which image dimensions should be
    multiplied by. Serves to change image size such that the patch dimension
    on which the network was trained fits reasonably into the input image.
    Of the form [horizontal_spacing, vertical_spacing, interlayer_spacing].
    If images are large, should have elements > 1
    If images are small should have elements < 1
    OUTPUTS
    preliminary mask - saved to input file directory - *_z_axis_prelim_mask.nii
    z axis corrected image - saved to input file directory - *_z_axis.nii
    intensity by slice image - saved to input file directory -
                            *_intensity_by_slice.png
    eroded preliminary mask - saved to input file directory -
                            *_eroded_mask.nii
    '''
    prelim_mask_fn = output_fn.split('_z_axis')[0] + '_z_axis_prelim_mask.nii'
    eroded_mask_fn = output_fn.split('_z_axis')[0] + '_eroded_mask.nii'
    print('Running inference for preliminary mask on: ' + input_fn)
    if frac_patch is None:
        if target_size is None:
            brain_seg_prediction(
                input_fn,
                prelim_mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                likelihood_categorization=likelihood_categorization)
        elif target_size is not None:
            brain_seg_prediction(
                input_fn,
                prelim_mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                target_size,
                likelihood_categorization=likelihood_categorization)
    if frac_patch is not None:
        if target_size is None:
            brain_seg_prediction(
                input_fn,
                prelim_mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                frac_patch=frac_patch,
                frac_stride=frac_stride,
                likelihood_categorization=likelihood_categorization)
        elif target_size is not None:
            brain_seg_prediction(
                input_fn,
                prelim_mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                target_size,
                frac_patch=frac_patch,
                frac_stride=frac_stride,
                likelihood_categorization=likelihood_categorization)

    eroded_mask_array, eroded_mask = \
        erode_img_by_slice(sitk.ReadImage(prelim_mask_fn),
                           kernel_radius=5)
    sitk.WriteImage(eroded_mask, eroded_mask_fn)

    source_img = sitk.ReadImage(input_fn)
    source_spacing = source_img.GetSpacing()
    source_array = sitk.GetArrayFromImage(source_img)

    # Get context from raw slices
    slice_intensity = []
    for j in range(0, source_array.shape[0]):
        current_roi = np.multiply(
            source_array[j, :, :], eroded_mask_array[j, :, :])
        current_slice_intensity = np.ma.masked_equal(current_roi, 0).mean()
        if np.amax(current_roi) == 0:  # missing data case
            current_slice_intensity = 0
        slice_intensity.append(current_slice_intensity)
    slice_intensity = np.array(slice_intensity)

    slice_ratio = slice_intensity / np.amax(slice_intensity)

    # Correct slice intensity
    corrected_slice_intensity = []
    for k in range(0, source_array.shape[0]):
        if slice_ratio[k] != 0:
            source_array[k, :, :] = source_array[k, :, :] / slice_ratio[k]
        current_roi = np.multiply(
            source_array[k, :, :], eroded_mask_array[k, :, :])
        current_slice_intensity = np.ma.masked_equal(current_roi, 0).mean()
        if np.amax(current_roi) == 0:
            current_slice_intensity = 0
        corrected_slice_intensity.append(current_slice_intensity)

    z_axis_corr_img = sitk.GetImageFromArray(source_array)
    z_axis_corr_img.SetSpacing(source_spacing)

    sitk.WriteImage(z_axis_corr_img, output_fn)

    # Build a visualization that will go into the source folder
    plot_intensity_comparison(slice_intensity,
                              corrected_slice_intensity,
                              output_fn.split('_z_axis')[0]
                              + '_intensity_by_slice.png')


def y_axis_correction(
        input_fn,
        output_fn,
        y_axis_mask):
    '''
    Function that handles y-axis corrections
    Y-axis corrections attempt to normalize image intensity variations within
    a slice. In particular, if there is an intensity gradient along the
    vertical axis of a given slice,y-axis correction will attempt to eliminate
    it. If specified, it is possible to mask the region which is to be
    corrected by Otsu binarization. This is highly recommended, as artifacts
    frequently appear.
    Y-axis correction is also called n4bias correction, and is handled by a
    SimpleITK filter
    '''
    source_img = sitk.ReadImage(input_fn, sitk.sitkFloat32)
    source_spacing = source_img.GetSpacing()
    source_direction = source_img.GetDirection()
    source_array = sitk.GetArrayFromImage(source_img)

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)

    n4b_corrector = sitk.N4BiasFieldCorrectionImageFilter()

    for i in range(0, source_array.shape[0]):
        current_layer_img = sitk.GetImageFromArray(source_array[i, :, :])
        if y_axis_mask:
            current_layer_mask = otsu_filter.Execute(current_layer_img)
            corrected_current_layer_img = n4b_corrector.Execute(
                current_layer_img, current_layer_mask)
        else:
            corrected_current_layer_img = n4b_corrector.Execute(
                current_layer_img)
        source_array[i, :, :] = sitk.GetArrayFromImage(
            corrected_current_layer_img)

    n4b_corrected_img = sitk.GetImageFromArray(source_array)
    n4b_corrected_img.SetSpacing(source_spacing)
    n4b_corrected_img.SetDirection(source_direction)

    sitk.WriteImage(n4b_corrected_img, output_fn)
