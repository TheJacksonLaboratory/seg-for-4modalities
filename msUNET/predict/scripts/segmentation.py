# author: Zachary Frohock
'''
Function that performs segmentation for a single file of .nii filetype
'''

from ..core.dice import dice_coef, dice_coef_loss
from ..core.utils import min_max_normalization, resample_img
from ..core.utils import remove_small_holes_and_points
from ..core.eval import out_LabelHot_map_2D
from tensorflow.keras.models import load_model
import SimpleITK as sitk


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

    seg_net = load_model(keras_paras.model_path,
                         compile=False,
                         custom_objects={'dice_coef_loss': dice_coef_loss,
                                         'dice_coef': dice_coef})

    imgobj = sitk.ReadImage(input_path)
    if target_size is None:
        resampled_imgobj = resample_img(imgobj,
                                        new_spacing=new_spacing,
                                        interpolator=sitk.sitkLinear)
    else:
        resampled_imgobj = resample_img(imgobj,
                                        interpolator=sitk.sitkLinear,
                                        target_size=target_size)
    img_array = sitk.GetArrayFromImage(resampled_imgobj)

    normed_array = min_max_normalization(img_array, normalization_mode)

    if frac_patch is None:
        out_label_img, out_likelihood_img = out_LabelHot_map_2D(
            normed_array,
            seg_net,
            pre_paras,
            keras_paras,
            likelihood_categorization=likelihood_categorization)
    else:
        out_label_img, out_likelihood_img = out_LabelHot_map_2D(
            normed_array,
            seg_net,
            pre_paras,
            keras_paras,
            frac_patch,
            frac_stride,
            likelihood_categorization=likelihood_categorization)

    out_label_img.CopyInformation(resampled_imgobj)
    out_likelihood_img.CopyInformation(resampled_imgobj)

    resampled_label_map = resample_img(
        out_label_img,
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

    resampled_label_map = remove_small_holes_and_points(resampled_label_map)

    sitk.WriteImage(resampled_label_map, output_path)
    sitk.WriteImage(resampled_likelihood_map,
                    output_path.split('.nii')[0] +
                    '_likelihood.nii')
