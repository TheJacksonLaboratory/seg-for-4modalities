# author: Zachary Frohock
'''
Function that performs segmentation for a single file of .nii filetype
'''

import os
from ..core.paras import PreParas, KerasParas
from ..core.dice import dice_coef, dice_coef_loss
from ..core.utils import min_max_normalization, resample_img
from ..core.eval import out_LabelHot_map_2D
from tensorflow.keras.models import load_model
from .. import __version__
import SimpleITK as sitk
import numpy as np
import argparse


def brain_seg_prediction(
        input_path,
        output_path,
        voxsize,
        pre_paras,
        keras_paras,
        new_spacing,
        normalization_mode,
        target_size=None,
        frac_patch=None,
        frac_stride=None,
        likelihood_categorization=False):
    # Function serving as an intermediatry between overall inference handler
    # and inference calculations themselves

    # load model
    seg_net = load_model(keras_paras.model_path,
                         compile=False,
                         custom_objects={'dice_coef_loss': dice_coef_loss,
                                         'dice_coef': dice_coef})

    imgobj = sitk.ReadImage(input_path)

    # re-sample to given voxel size
    if target_size is None:
        resampled_imgobj = resample_img(imgobj,
                                        new_spacing=new_spacing,
                                        # new_spacing=[0.08, 0.08, 0.76], # 2d
                                        # images [1,1,1]
                                        interpolator=sitk.sitkLinear)
    if target_size is not None:
        resampled_imgobj = resample_img(imgobj,
                                        interpolator=sitk.sitkLinear,
                                        target_size=target_size)

    img_array = sitk.GetArrayFromImage(resampled_imgobj)
    normed_array = min_max_normalization(img_array, normalization_mode)
    if frac_patch is None:
        out_label_map, out_likelihood_map = out_LabelHot_map_2D(normed_array,
                                                                seg_net,
                                                                pre_paras,
                                                                keras_paras,
                                                                likelihood_categorization=likelihood_categorization)
    if frac_patch is not None:
        out_label_map, out_likelihood_map = out_LabelHot_map_2D(normed_array,
                                                                seg_net,
                                                                pre_paras,
                                                                keras_paras,
                                                                frac_patch,
                                                                frac_stride,
                                                                likelihood_categorization=likelihood_categorization)

    out_label_img = sitk.GetImageFromArray(out_label_map.astype(np.uint8))
    out_likelihood_img = sitk.GetImageFromArray(
        out_likelihood_map.astype(np.float32))
    out_label_img.CopyInformation(resampled_imgobj)
    out_likelihood_img.CopyInformation(resampled_imgobj)

    resampled_label_map = resample_img(out_label_img,
                                       new_spacing=imgobj.GetSpacing(),
                                       target_size=imgobj.GetSize(),
                                       interpolator=sitk.sitkNearestNeighbor,
                                       revert=True)
    resampled_likelihood_map = resample_img(
        out_likelihood_img,
        new_spacing=imgobj.GetSpacing(),
        target_size=imgobj.GetSize(),
        interpolator=sitk.sitkNearestNeighbor,
        revert=True)

    # Fill small holes in binary mask
    binary_hole_filler = sitk.BinaryFillholeImageFilter()
    binary_inversion_filter = sitk.InvertIntensityImageFilter()
    binary_inversion_filter.SetMaximum(1)
    missing_hole_fill = binary_hole_filler.Execute(resampled_label_map)
    missing_hole_fill_invert = binary_inversion_filter.Execute(missing_hole_fill)
    isolated_brain_invert = binary_hole_filler.Execute(missing_hole_fill_invert)
    resampled_label_map = binary_inversion_filter.Execute(isolated_brain_invert)

    sitk.WriteImage(resampled_label_map, output_path)
    sitk.WriteImage(
        resampled_likelihood_map,
        output_path.split('.nii')[0] +
        '_likelihood.nii')
