# author: Zachary Frohock
'''
Functions handling pre-inference image processing
'''

import shutil
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from ..scripts.segmentation import brain_seg_prediction


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
    # Function that handles performing z-axis correction
    # Z-axis correction attempts to normalize the intensity of brain regions across slices. To do so,
    # it calculates a preliminary mask on raw data. That preliminary mask is eroded to increase the likelihood
    # that brain tissue is all that is contained within. The raw image intensity in that region is then calculated,
    # and used to calculate a factor by which each slice should be multiplied to normalize brain region intensity.
    # INPUTS:
    # input_fn: full file path of raw data source
    # output_fn: full file path of output z-axis corrected data
    # voxsize: size to which 3d voxels should be resampled. Generally kept at 0.1 for consistency
    # pre_paras: Set of parameters to do with image patching
    # keras_paras: Set of parameters to do with inference, most relevant is classification threshold thd
    # new_spacing: Muliplicative factor by which image dimensions should be multiplied by. Serves to
    # change image size such that the patch dimension on which the network was trained fits reasonably
    # into the input image. Of the form [horizontal_spacing, vertical_spacing, interlayer_spacing].
    # If images are large, should have elements > 1
    # If images are small should have elements < 1
    # OUTPUTS
    # preliminary mask - saved to input file directory - *_z_axis_prelim_mask.nii
    # z axis corrected image - saved to input file directory - *_z_axis.nii
    # intensity by slice image - saved to input file directory - *_intensity_by_slice.png
    # eroded preliminary mask - saved to input file directory -
    # *_eroded_mask.nii

    # First run inference to get a preliminary mask
    # We assume that the mask will be centered on some brain tissue, it just
    # may not cover the whole thing.
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

    # Load and erode the preliminary mask
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(5)

    prelim_mask = sitk.ReadImage(prelim_mask_fn)
    prelim_mask_spacing = prelim_mask.GetSpacing()
    prelim_mask_array = sitk.GetArrayFromImage(prelim_mask)

    # Do the erosion by layer, as eroding the 3d image tends to eliminate any
    # signal in the first/last slices
    for i in range(0, prelim_mask_array.shape[0]):
        current_layer_img = sitk.GetImageFromArray(prelim_mask_array[i, :, :])
        current_layer_img.SetSpacing(prelim_mask_spacing)
        current_layer_img = erode_filter.Execute(current_layer_img)
        prelim_mask_array[i, :, :] = sitk.GetArrayFromImage(current_layer_img)

    # Grab the eroded mask. We'll save that in the source folder
    eroded_mask_array = prelim_mask_array
    eroded_mask = sitk.GetImageFromArray(eroded_mask_array)

    # Load in the source image to do z-axis correction
    source_img = sitk.ReadImage(input_fn)
    source_spacing = source_img.GetSpacing()
    source_array = sitk.GetArrayFromImage(source_img)

    # Look at the intensity of the source image in the ROI determined by prelim mask
    # Determine the max mean intensity, normalize all other slices to that
    # same intensity
    slice_intensity = []
    for j in range(0, source_array.shape[0]):
        current_roi = np.multiply(
            source_array[j, :, :], eroded_mask_array[j, :, :])
        current_slice_intensity = 0
        current_slice_intensity = np.ma.masked_equal(current_roi, 0).mean()
        # Images can occasionally have no data in a given slice. Normally occurs in first and last
        # If this is the case, just set the intensity to zero because the ROI
        # is of zero size
        if np.amax(current_roi) == 0:
            current_slice_intensity = 0
        slice_intensity.append(current_slice_intensity)

    slice_intensity = np.array(slice_intensity)

    # Calculate the factors we will use to divide slice intensity by
    slice_ratio = slice_intensity / np.amax(slice_intensity)

    corrected_slice_intensity = []
    # Perform intensity normalization by slice
    for k in range(0, source_array.shape[0]):
        # Do not try to correct a slice with zero ROI, it will lead to infinite
        # intensities
        if slice_ratio[k] != 0:
            source_array[k, :, :] = source_array[k, :, :] / slice_ratio[k]
        current_roi = np.multiply(
            source_array[k, :, :], eroded_mask_array[k, :, :])
        current_slice_intensity = 0
        current_slice_intensity = np.ma.masked_equal(current_roi, 0).mean()
        # Keep track of the corrected ROI intensity to provide a pleasant
        # visualization
        if np.amax(current_roi) == 0:
            current_slice_intensity = 0
        corrected_slice_intensity.append(current_slice_intensity)

    z_axis_corr_img = sitk.GetImageFromArray(source_array)
    z_axis_corr_img.SetSpacing(source_spacing)

    sitk.WriteImage(z_axis_corr_img, output_fn)

    # Build a visualization that will go into the source folder
    plt.figure()
    plt.plot(slice_intensity)
    plt.plot(corrected_slice_intensity)
    plt.xlabel('Slice Index')
    plt.ylabel('Mean Intensity in Preliminary Mask Region')
    plt.title('ROI Intensity - Z-Axis Correction')
    plt.legend(['Source Image', 'Z-Axis Corrected'])
    plt.savefig(output_fn.split('_z_axis')[0] + '_intensity_by_slice.png')

    eroded_mask.SetSpacing(prelim_mask_spacing)

    sitk.WriteImage(eroded_mask, eroded_mask_fn)


def y_axis_correction(
        input_fn,
        output_fn,
        voxsize,
        pre_paras,
        keras_paras,
        new_spacing,
        y_axis_mask):
    # Function that handles y-axis corrections
    # Y-axis corrections attempt to normalize image intensity variations within a slice.
    # In particular, if there is an intensity gradient along the vertical axis of a given slice,
    # y-axis correction will attempt to eliminate it. If specified, it is possible to mask the
    # region which is to be corrected by Otsu binarization. This is highly recommended, as artifacts
    # frequently appear.
    # Y-axis correction is also called n4bias correction, and is handled by a
    # SimpleITK filter
    source_img = sitk.ReadImage(input_fn, sitk.sitkFloat32)
    source_spacing = source_img.GetSpacing()
    source_direction = source_img.GetDirection()
    source_array = sitk.GetArrayFromImage(source_img)

    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)

    n4b_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4

    for i in range(0, source_array.shape[0]):
        current_layer_img = sitk.GetImageFromArray(source_array[i, :, :])
        if y_axis_mask:
            current_layer_mask = otsu_filter.Execute(current_layer_img)
            corrected_current_layer_img = n4b_corrector.Execute(
                current_layer_img, current_layer_mask)
        if not y_axis_mask:
            corrected_current_layer_img = n4b_corrector.Execute(
                current_layer_img)
        source_array[i, :, :] = sitk.GetArrayFromImage(
            corrected_current_layer_img)

    n4b_corrected_img = sitk.GetImageFromArray(source_array)
    n4b_corrected_img.SetSpacing(source_spacing)
    n4b_corrected_img.SetDirection(source_direction)

    sitk.WriteImage(n4b_corrected_img, output_fn)
