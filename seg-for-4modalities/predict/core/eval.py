# Copyright The Jackson Laboratory, 2022
# author: Zachary Frohock
# Built from framework of https://github.com/CAMRIatUNC/RodentMRISkullStripping
'''
Function that runs inference on provided array against segmentation model
'''

import SimpleITK as sitk
import numpy as np
from .utils import dim_2_categorical


def out_LabelHot_map_2D(
        img,
        seg_net,
        pre_paras,
        keras_paras,
        frac_patch=None,
        frac_stride=None,
        likelihood_categorization=False):
    '''
    Controls image patching and inference on those patches
    Parameters
    ----------
    img: array like (_, _, _)
        Processed image to patch and infer
    seg_net: Loaded Keras model
        Keras U-Net segmentation model
    pre_paras: Class
        Class containing image processing parameters: patch dims, patch stride
    keras_paras: Class
        Class containing keras parameters for inference, including model path,
        threshold, and image format.
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
    label_img: SimpleITK object
        Final binarized mask
    likelihood_img: SimpleITK object
        Final likelihood mask that is binarized to create final_mask
    '''
    patch_dims = [0, 0, 0]
    label_dims = [0, 0, 0]
    strides = [0, 0, 0]
    n_class = pre_paras.n_class
    length, col, row = img.shape
    if frac_patch is None:
        patch_dims = pre_paras.patch_dims
        label_dims = pre_paras.patch_label_dims
        strides = pre_paras.patch_strides
    else:
        img_shape = img.shape
        patch_dims = pre_paras.patch_dims
        label_dims = pre_paras.patch_label_dims
        strides[0] = pre_paras.patch_strides[0]
        strides[1] = int(np.floor(img_shape[1] * frac_stride))
        strides[2] = int(np.floor(img_shape[2] * frac_stride))

    if (img.shape[0] < patch_dims[0]) or (img.shape[1] < patch_dims[1]):
        raise ValueError("The values for patch dimensions are incompatible \
            with resampled image dimensions. The value of the patch \
            dimension is: {0}, and the value of the image dimension is: {1}\
            . Generally this error appears when the image dimensions are \
            too small, meaning no patches can be generated. If using target \
            size, increase the target dimensions. If not, decrease the value \
            of in-slice spacing values.")

    categorical_map = np.zeros((n_class, length, col, row), dtype=np.uint8)
    likelihood_map = np.zeros((length, col, row), dtype=np.float32)
    likelihood_map_max = np.zeros((length, col, row), dtype=np.float32)
    counter_map = np.zeros((length, col, row), dtype=np.float32)
    length_step = int(patch_dims[0] / 2)

    # Build lists of indicies for patch slicing in 3D
    slice_range_forward = list(
        range(0, length - patch_dims[0] + 1, strides[0]))
    col_range_forward = list(
        range(0, col - patch_dims[1] + 1, strides[1]))
    col_range_forward.append(
        col_range_forward[-1] +
        (col-(patch_dims[1]+(len(col_range_forward)-1)*strides[1])))
    row_range_forward = list(
        range(0, row - patch_dims[2] + 1, strides[2]))
    row_range_forward.append(
        row_range_forward[-1] +
        (row-(patch_dims[2]+(len(row_range_forward)-1)*strides[2])))
    slice_range_backward = list(
        range(length, patch_dims[0] - 1, -strides[0]))
    col_range_backward = list(
        range(col, patch_dims[1] - 1, -strides[1]))
    col_range_backward.append(
        col_range_backward[-1] -
        (col-(patch_dims[1]+(len(col_range_backward)-1)*strides[1])))
    row_range_backward = list(
        range(row, patch_dims[2] - 1, -strides[2]))
    row_range_backward.append(
        row_range_backward[-1] -
        (row-(patch_dims[2]+(len(row_range_backward)-1)*strides[2])))

    for i in slice_range_forward:
        for j in col_range_forward:
            for k in row_range_forward:
                cur_patch = img[i:i + patch_dims[0],
                                j:j + patch_dims[1],
                                k:k + patch_dims[2]][:].reshape([1,
                                                                patch_dims[0],
                                                                patch_dims[1],
                                                                patch_dims[2]])
                if keras_paras.img_format == 'channels_last':
                    cur_patch = np.transpose(cur_patch, (0, 2, 3, 1))

                cur_patch_output = seg_net.predict(
                    cur_patch, batch_size=1, verbose=0)

                if isinstance(cur_patch_output, list):
                    cur_patch_output = cur_patch_output[keras_paras.outID]

                cur_patch_output = np.squeeze(cur_patch_output)
                cur_patch_out_label = cur_patch_output.copy()
                cur_patch_out_label = np.where(
                    cur_patch_output > keras_paras.thd, 1, 0)

                middle = i + length_step
                cur_patch_out_label = dim_2_categorical(
                    cur_patch_out_label, n_class)

                categorical_map[
                    :,
                    middle,
                    j:j + label_dims[1],
                    k:k + label_dims[2]] = categorical_map[
                                                :,
                                                middle,
                                                j:j + label_dims[1],
                                                k:k + label_dims[2]] + cur_patch_out_label
                likelihood_map[
                    middle,
                    j:j + label_dims[1],
                    k:k + label_dims[2]] = likelihood_map[
                                                middle,
                                                j:j + label_dims[1],
                                                k:k + label_dims[2]] + cur_patch_output
                likelihood_map_max[
                    middle,
                    j:j + label_dims[1],
                    k:k + label_dims[2]] = np.maximum(
                                                likelihood_map_max[
                                                    middle,
                                                    j:j + label_dims[1],
                                                    k:k + label_dims[2]], cur_patch_output)
                counter_map[middle,
                            j:j + label_dims[1],
                            k:k + label_dims[2]] += 1

    for i in slice_range_backward:
        for j in col_range_backward:
            for k in row_range_backward:
                cur_patch = img[i - patch_dims[0]:i,
                                j - patch_dims[1]:j,
                                k - patch_dims[2]:k][:].reshape([1,
                                                                patch_dims[0],
                                                                patch_dims[1],
                                                                patch_dims[2]])
                if keras_paras.img_format == 'channels_last':
                    cur_patch = np.transpose(cur_patch, (0, 2, 3, 1))

                cur_patch_output = seg_net.predict(
                    cur_patch, batch_size=1, verbose=0)

                if isinstance(cur_patch_output, list):
                    cur_patch_output = cur_patch_output[keras_paras.outID]
                cur_patch_output = np.squeeze(cur_patch_output)

                cur_patch_out_label = cur_patch_output.copy()
                cur_patch_out_label = np.where(
                    cur_patch_output > keras_paras.thd, 1, 0)

                middle = i - patch_dims[0] + length_step
                cur_patch_out_label = dim_2_categorical(
                    cur_patch_out_label, n_class)
                categorical_map[
                    :,
                    middle,
                    j - label_dims[1]:j,
                    k - label_dims[2]:k] = categorical_map[
                                                :,
                                                middle,
                                                j - label_dims[1]:j,
                                                k - label_dims[2]:k] + cur_patch_out_label
                likelihood_map[
                    middle,
                    j - label_dims[1]:j,
                    k - label_dims[2]:k] = likelihood_map[
                                                middle,
                                                j - label_dims[1]:j,
                                                k - label_dims[2]:k] + cur_patch_output
                likelihood_map_max[
                    middle,
                    j - label_dims[1]:j,
                    k - label_dims[2]:k] = np.maximum(
                                                likelihood_map_max[
                                                    middle,
                                                    j - label_dims[1]:j,
                                                    k - label_dims[2]:k], cur_patch_output)
                counter_map[middle,
                            j - label_dims[1]:j,
                            k - label_dims[2]:k] += 1

    if likelihood_categorization == False:
        likelihood_map = likelihood_map_max
        label_map = np.where(likelihood_map > keras_paras.thd, 1, 0)
    else:
        likelihood_map = np.divide(likelihood_map, counter_map)
        label_map = np.where(likelihood_map > keras_paras.thd, 1, 0)

    label_img = sitk.GetImageFromArray(label_map.astype(np.uint8))
    likelihood_img = sitk.GetImageFromArray(likelihood_map.astype(np.float32))

    return label_img, likelihood_img
