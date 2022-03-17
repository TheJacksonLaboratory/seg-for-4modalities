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
        output_orientation,
        target_size=None,
        frac_patch=None,
        frac_stride=None,
        likelihood_categorization=False):
    '''
    Handled segmentation for single image. Segmentation only.
    Parameters
    ----------
    input_path: String
        Path to processed input file
    output_path: String
        Path to write mask to
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
    Output
    ----------
    final_mask:
        Final binary mask from segmentation; written to disk
    likelihood_mask:
        Final likelihood mask that is binarized to create final_mask;
        written to disk
    '''
    seg_net = load_model(keras_paras.model_path,
                         compile=False,
                         custom_objects={'dice_coef_loss': dice_coef_loss,
                                         'dice_coef': dice_coef})

    imgobj = sitk.DICOMOrient(sitk.ReadImage(input_path),'LPS')
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
    resampled_label_map = sitk.Cast(resampled_label_map,
                                    imgobj.GetPixelIDValue())

    sitk.WriteImage(sitk.DICOMOrient(resampled_label_map, output_orientation),
                    output_path)
    sitk.WriteImage(sitk.DICOMOrient(resampled_likelihood_map,
                                     output_orientation),
                    output_path.split('.nii')[0] +
                    '_likelihood.nii')
