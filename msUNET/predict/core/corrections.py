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
        output_orientation,
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
    Parameters
    ----------
    input_fn: String
        Full file path of raw data source
    output_fn: String
        Full file path of output z-axis corrected data
    voxsizse: Float
        Size of voxels in mm
    pre_paras: Class
        Class containing image processing parameters: patch dims, patch stride
    keras_paras: Class
        Class containing keras parameters for inference, including model path,
        threshold, and image format.
    new_spacing: array-like, (_, _, _)
        Spacing to which an image will be resampled for inference. First two
        entries correspond to in-slice dimensions, third between slices.
    normalization_mode: String
        To perform normalization 'by_img' or 'by_slice' before inference
    target_size: array like (_, _, _) or None
        Dimensions to which all images will be sampled prior to patching.
        If None, images are not resampled to a constant size, and new_spacing
        is used to determine image resampling.
    frac_patch: Float in range (0, 1) or None
        Fraction of resampled image dimensions the patch size should be set to.
        If None, use fixed values from pre_paras.patch_dims
    frac_stride: Float in range (0,1) or None
        Fraction of resampled image dimensions patch stride should be set to.
        If None, use fixed values from per_paras.patch_stride
    likelihood_categorization: Bool
        How should final binarization of score -> mask be done. If True, use
        the max value of likelihood per-pixel. If False, use the mean value.
    Outputs
    ----------
    preliminary mask
        Mask calculated from first pass of inference; write to disk
    z axis corrected image
        Image after z-axis corrections; write to disk
    intensity by slice
        Plot of ROI intensity in source and z-axis corrected images by slice;
        written to disk
    eroded preliminary mask
        Eroded mask from preliminary inference pass; written to disk
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
                output_orientation,
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
                output_orientation,
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
                output_orientation,
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
                output_orientation,
                target_size,
                frac_patch=frac_patch,
                frac_stride=frac_stride,
                likelihood_categorization=likelihood_categorization)

    eroded_mask_array, eroded_mask = \
        erode_img_by_slice(sitk.ReadImage(prelim_mask_fn),
                           kernel_radius=5)
    sitk.WriteImage(sitk.DICOMOrient(eroded_mask, output_orientation),
                    eroded_mask_fn)

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

    sitk.WriteImage(sitk.DICOMOrient(z_axis_corr_img, output_orientation),
                    output_fn)

    # Build a visualization that will go into the source folder
    plot_intensity_comparison(slice_intensity,
                              corrected_slice_intensity,
                              output_fn.split('_z_axis')[0]
                              + '_intensity_by_slice.png')


def y_axis_correction(
        input_fn,
        output_fn,
        y_axis_mask,
        output_orientation):
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
    Parameters
    ----------
    input_fn: String
        Full file path of raw data source
    output_fn: String
        Full file path of output y-axis corrected data
    y_axis_mask: Bool
        Whether to apply n4bias field correction to the entire image (False),
        or an binary otsu masked region (True).
    Outputs
    n4b corrected image
        Source image after n4b (y-axis) corrections
    ----------
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

    sitk.WriteImage(sitk.DICOMOrient(n4b_corrected_img, output_orientation),
                    output_fn)
