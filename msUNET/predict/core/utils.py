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

def min_max_normalization(img, normalization_mode='by_img'):
    # Function that normalizes input data.
    # INPUTS
    # img: 3D numpy array containing image data for a single MRI scan
    # by_slice: Boolean dictating whether normalization should be done by-image (0) or by slice (1)
    # By slice normalization generally eliminates the need for z-axs correction. Ensure that the model
    # you are planning to use has been trained using data that has been normalized in the same way.
    # OUTPUTS
    # new_img: Input image normalized to range [0,1]

    # By image normalization
    if normalization_mode == 'by_img':
        new_img = img.copy()
        new_img = new_img.astype(np.float32)

        min_val = np.min(new_img)
        max_val = np.max(new_img)
        new_img = (np.asarray(new_img).astype(
            np.float32) - min_val) / (max_val - min_val)

    # By slice normalization
    if normalization_mode == 'by_slice':
        new_img = img.copy()
        new_img = img.astype(np.float32)
        for slice_index in range(0, new_img.shape[0]):
            min_val = np.amin(new_img[slice_index, :, :])
            max_val = np.amax(new_img[slice_index, :, :])
            new_img[slice_index, :, :] = (new_img[slice_index, :, :].astype(
                np.float32) - min_val) / (max_val - min_val)

    return new_img


def dim_2_categorical(label, num_class):
    # Deprecated function intended to assist in conversion of data with
    # additional input channels
    dims = label.ndim
    if dims == 2:
        col, row = label.shape
        ex_label = np.zeros((num_class, col, row))
        for i in range(0, num_class):
            ex_label[i, ...] = np.asarray(label == i).astype(np.uint8)
    elif dims == 3:
        leng, col, row = label.shape
        ex_label = np.zeros((num_class, leng, col, row))
        for i in range(0, num_class):
            ex_label[i, ...] = np.asarray(label == i).astype(np.uint8)
    else:
        raise Exception
    return ex_label


def resample_img(
        imgobj,
        new_spacing=None,
        interpolator=sitk.sitkLinear,
        target_size=None,
        revert=False):
    # Function that resamples input images for compatibility with inference model
    # INPTS
    # imgobj: SimpleITK image object corresponding to raw data
    # new_spacing: Muliplicative factor by which image dimensions should be multiplied by. Serves to
    # change image size such that the patch dimension on which the network was trained fits reasonably
    # into the input image. Of the form [horizontal_spacing, vertical_spacing, interlayer_spacing].
    # If images are large, should have elements > 1
    # If images are small should have elements < 1
    # Interpolator: function used to interpolate in the instance new_spacing != [1,1,1]. Default is nearest
    # neighbor as to not introduce new values
    # New_size: declared size for new images. If left as None will calculate the new size automatically
    # baseon on new spacing and old image dimensions/spacing
    # OUTPUTS
    # resampled_imgobj: SimpleITK image object resampled in the desired manner
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(imgobj.GetDirection())
    resample.SetOutputOrigin(imgobj.GetOrigin())
    if not revert:
        if target_size is None:
            orig_img_spacing = np.array(imgobj.GetSpacing())
            resample.SetOutputSpacing(new_spacing)
            orig_size = np.array(imgobj.GetSize(), dtype=np.int)
            orig_spacing = np.array(imgobj.GetSpacing())
            target_size = orig_size * (orig_spacing / new_spacing)
            target_size = np.ceil(target_size).astype(
                np.int)  # Image dimensions are in integers
            target_size = [int(s) for s in target_size]
            new_spacing[2] = orig_img_spacing[2]

            target_size_final = target_size

        if target_size is not None:
            new_spacing = [0, 0, 0]
            orig_img_dims = np.array(sitk.GetArrayFromImage(imgobj).shape)
            orig_img_spacing = np.array(imgobj.GetSpacing())
            target_size[2] = orig_img_dims[0]
            target_size[1] = int(
                np.floor(
                    (target_size[0] /
                     orig_img_dims[1]) *
                    orig_img_dims[2]))
            spacing_ratio_1 = target_size[0] / orig_img_dims[1]
            spacing_ratio_2 = target_size[1] / orig_img_dims[2]
            new_spacing[0] = orig_img_spacing[0] / spacing_ratio_1  # orig 1
            new_spacing[1] = orig_img_spacing[1] / spacing_ratio_2  # orig 2
            new_spacing[2] = orig_img_spacing[2]
            resample.SetOutputSpacing(new_spacing)

            # Correct target size image dimensions
            target_size_final = [0, 0, 0]
            target_size_final[0] = target_size[1]
            target_size_final[1] = target_size[0]
            target_size_final[2] = target_size[2]

    if revert:
        resample.SetOutputSpacing(new_spacing)
        target_size_final = target_size

    resample.SetSize(np.array(target_size_final, dtype='int').tolist())

    resampled_imgobj = resample.Execute(imgobj)
    return resampled_imgobj


def listdir_nohidden(path):
    # Function that lists the full paths of files and directories in path.
    # Does not list hidden files.
    return glob.glob(os.path.join(path, '*'))


def get_suffix(z_axis, y_axis):
    # Function that determines the suffix to be used in finding the file in a modality folder on which
    # final inference is to be run
    # INPUTS
    # z_axis: Was z-axis correction used, string 'True' if so
    # y_axis: Was y_axis correction used, string 'True' if so
    # OUTPUTS
    # suffix: file suffix to be appended to source data filename for final
    # inference
    suffix = ''
    if z_axis == 'True':
        suffix = '_z_axis'
    if y_axis == 'True':
        suffix = suffix + '_n4b'
    return suffix


def low_snr_check(source_array, source_fn, low_snr_threshold):
    # Function that determines whether a raw data slice should be flagged for manual review due to a low
    # signal to noise ratio. Does so by comparing mean intensity in center of image to mean intensity
    # in corners. Assumes that the brain is roughly centered in the image
    # INPUTS
    # source_array: numpy array corresponding to entire MRI image scan
    # source_fn: full path to source file
    # low_snr_threshold: Multiplicative factor below which slice will be flagged
    # OUTPUTS
    # snr_check_list: list of slices and the corresponding files that contain them that have been flagged for
    # TODO: Use aim 1 masks to define 'signal' region instead of assuming
    # circle at center
    snr_check_list = []
    for slice_index in range(0, source_array.shape[0]):
        current_slice = np.array(source_array[slice_index, :, :])
        i, j = np.indices(current_slice.shape)
        # Grab the mean of a 3x3 circle in the center of the image
        center_mean = np.array(current_slice[((i -
                                               (current_slice.shape[0] //
                                                2))**2 < 9) & ((j -
                                                                (current_slice.shape[1] //
                                                                 2))**2 < 9)]).mean()
        k, l = 3, 3
        # Grab the mean of a 3x3 box on the upper left hand corner of the image
        edge_mean = current_slice[max(
            0, k - 3):k + 3, max(0, l - 3):l + 3].mean()
        # Compare the center mean (brain) to the edge mean (background). If
        # they are very different, warn
        if center_mean <= edge_mean * low_snr_threshold:
            snr_check_list.append(
                source_fn + ' -- Slice: ' + str(slice_index + 1) + ' -- LOW SNR WARNING')

    return snr_check_list


def mask_area_check(mask_array, source_fn, source_array):
    # Function that determines whether a raw data slice should be flagged for manual review due to either a
    # low or high mask area. If the percentage of pixels in classified as brain in a given slice is outside
    # the interval [0.04,0.8], a flag will be raised.
    low_mask_area_check_list = []
    high_mask_area_check_list = []
    for slice_index in range(0, mask_array.shape[0]):
        current_slice = np.array(mask_array[slice_index, :, :])
        current_slice_source = np.array(source_array[slice_index, :, :])
        total_pixels = current_slice.size
        mask_pixels = (np.asarray(current_slice) > 0).sum()
        source_data_pixels = (
            np.asarray(current_slice_source) > current_slice_source.mean()).sum()
        mask_ratio = mask_pixels / total_pixels
        source_data_ratio = source_data_pixels / total_pixels
        if mask_ratio < 0.04 and source_data_ratio > 0.04:
            low_mask_area_check_list.append(
                source_fn + ' -- Slice: ' + str(slice_index + 1) + ' -- LOW MASK AREA WARNING')
        if mask_ratio > 0.6 and source_data_ratio < 0.8:
            high_mask_area_check_list.append(
                source_fn + ' -- Slice: ' + str(slice_index + 1) + ' -- HIGH MASK AREA WARNING')

    return low_mask_area_check_list, high_mask_area_check_list


def intermediate_likelihood_check(likelihood_array, source_fn, source_array):
    # Function that determines whether a raw data slice should be flagged for manual review due to a higher
    # than expected percentage of pixels having a score between 0.1 and 0.75. If more than 3% of a slice's
    # pixels have score values in this intermediate range, the slice is
    # flagged.
    high_int_likelihood_check = []
    for slice_index in range(0, likelihood_array.shape[0]):
        current_slice = np.array(likelihood_array[slice_index, :, :])
        total_pixels = current_slice.size
        confident_mask_pixels = (current_slice > 0.75).sum()
        confident_back_pixels = (current_slice < 0.1).sum()
        intermediate_pixels = total_pixels - confident_back_pixels - confident_mask_pixels
        intermediate_pixel_ratio = intermediate_pixels / total_pixels
        if intermediate_pixel_ratio > 0.03:
            high_int_likelihood_check.append(
                source_fn + ' -- Slice: ' + str(slice_index + 1) + ' -- LOW PREDICTION BORDER CONFIDENCE')

    return high_int_likelihood_check


def convex_hull_image(data):
    # Function that calculates and draws the convex hull for a 2D binary image
    region = np.argwhere(data)
    hull = ConvexHull(region)
    verts = [(region[v, 0], region[v, 1]) for v in hull.vertices]
    img = Image.new('L', data.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T


def solidity_check(mask_array, source_fn, source_array):
    # Function that determines whether a mask should be flagged for manual review due to having a low solidity.
    # For each slice, the percentage of pixels classified as brain is compared to the number of pixels
    # contained in the convex hull containing those pixels. If the count of pixels in the brain is less than
    # 90% of the count in the convex hull, a flag is thrown.
    solidity_check = []
    for slice_index in range(0, mask_array.shape[0]):
        current_slice = np.asarray(mask_array[slice_index, :, :])
        # Convex hull operation fails if there are fewer than 3 pixels classified as brain
        # In this case, this check is not relevant. Zero-pixel masks are caught
        # by other checks.
        try:
            current_slice_convex_hull = convex_hull_image(current_slice)
        except BaseException:
            current_slice_convex_hull = current_slice
        total_pixels = current_slice.size
        mask_pixels = (current_slice == 1).sum()
        convex_hull_pixels = (current_slice_convex_hull == 1).sum()
        slice_solidity = mask_pixels / convex_hull_pixels
        if slice_solidity < 0.9:
            solidity_check.append(
                source_fn + ' -- Slice: ' + str(slice_index + 1) + ' -- LOW MASK SOLIDITY')
    return solidity_check


def str2bool(v):
    # A function that converts a collection of possible input values corresponding to True and False from
    # strings too booleans
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
