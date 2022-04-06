# author: Zachary Frohock
'''
Functions handling orientation detection and re-slicing
'''

import SimpleITK as sitk
import numpy as np
import scipy.ndimage as snd


def get_orientation_string(img):
    '''
    Function that gets a three character string corresponding to the orientation of
    the input image
    Parameters
    ----------
    img: array like, sitk object (_, _, _)
        3D SimpleITK object, must have been read from a .nii(.gz) file
    Outputs
    ----------
    orientation: string, 3 characters
        Orientation of input image
    '''
    img_array = sitk.GetArrayFromImage(img)
    if len(img_array.shape) > 3:
        img_4d = img
        img_array = img_array[0, :, :, :]
        img = sitk.GetImageFromArray(img_array)
        img.CopyInformation(img_4d[:, :, :, 0])
    orientationFilter = sitk.DICOMOrientImageFilter()
    input_orientation = orientationFilter.GetOrientationFromDirectionCosines(
        img.GetDirection())

    return input_orientation


def flip_vertically(sitk_obj):
    '''
    Function that flips a 3D image such that each slice is flipped vertically
    Parameters
    ----------
    sitk_obj: array like, sitk object (_, _, _)
        3D SimpleITK object, must have been read from a .nii(.gz) file
    Outputs
    ----------
    Flipped sitk object
    '''
    return sitk.Flip(sitk_obj, [False, True, False])


def rotate_90(sitk_obj):
    '''
    Function that rotates a 3D image 90 degrees counterclockwise slice-wise
    Parameters
    ----------
    sitk_obj: array like, sitk object (_, _, _)
        3D SimpleITK object, must have been read from a .nii(.gz) file
    Outputs
    ----------
    Rotated sitk object
    '''
    permutation_filter = sitk.PermuteAxesImageFilter()
    permutation_filter.SetOrder((1, 0, 2))

    return permutation_filter.Execute(sitk_obj)


def saggital_to_axial(sitk_obj, long_axis, flip_vertical):
    '''
    Function that reslices MRI scans with saggital slices to axial slices
    Parameters
    ----------
    sitk_obj: array like, sitk object (_, _, _)
        3D SimpleITK object, must have been read from a .nii(.gz) file.
        Must have saggital slices.
    long_axis: string, choices = ['horizontal', 'vertical']
        Slice-wise axis to which the long edge of the brain is aligned
    flip_vertical: bool
        Whether to flip the output axial image vertically
    Outputs
    ----------
    axial_obj: array_like, sitk object (_, _, _)
        3D SimpleITK object, resliced to axial plane
    '''
    permutation_filter = sitk.PermuteAxesImageFilter()
    saggital_obj = sitk_obj
    if long_axis == 'vertical':
        permutation_filter.SetOrder((2, 0, 1))
    if long_axis == 'horizontal':
        permutation_filter.SetOrder((2, 1, 0))
    axial_obj = permutation_filter.Execute(saggital_obj)
    if flip_vertical == True:
        axial_obj = sitk.Flip(axial_obj, [False, True, False])    
    axial_obj = sitk.Flip(axial_obj, [False, False, True])

    return axial_obj


def coronal_to_axial(sitk_obj, long_axis, flip_vertical):
    '''
    Function that reslices MRI scans with coronal slices to axial slices
    Parameters
    ----------
    sitk_obj: array like, sitk object (_, _, _)
        3D SimpleITK object, must have been read from a .nii(.gz) file.
        Must have coronal slices.
    long_axis: string, choices = ['horizontal', 'vertical']
        Slice-wise axis to which the long edge of the brain is aligned
    flip_vertical: bool
        Whether to flip the output axial image vertically
    Outputs
    ----------
    axial_obj: array_like, sitk object (_, _, _)
        3D SimpleITK object, resliced to axial plane
    '''
    permutation_filter = sitk.PermuteAxesImageFilter()
    coronal_obj = sitk_obj
    if long_axis == 'vertical':
        permutation_filter.SetOrder((0, 2, 1))
    if long_axis == 'horizontal':
        permutation_filter.SetOrder((1, 2, 0))
    axial_obj = permutation_filter.Execute(coronal_obj)
    if flip_vertical == True:
        axial_obj = sitk.Flip(axial_obj, [False, True, False])
    axial_obj = sitk.Flip(axial_obj, [False, False, True])

    return axial_obj


def axial_to_coronal(sitk_obj, long_axis, flip_vertical):
    '''
    Function that reslices MRI scans with axial slices to coronal slices
    Parameters
    ----------
    sitk_obj: array like, sitk object (_, _, _)
        3D SimpleITK object, must have been read from a .nii(.gz) file.
        Must have axial slices.
    long_axis: string, choices = ['horizontal', 'vertical']
        Slice-wise axis to which the long edge of the brain is aligned
    flip_vertical: bool
        Whether to flip the input axial image vertically
    Outputs
    ----------
    axial_obj: array_like, sitk object (_, _, _)
        3D SimpleITK object, resliced to coronal plane
    '''
    permutation_filter = sitk.PermuteAxesImageFilter()
    axial_obj = sitk_obj
    axial_obj = sitk.Flip(axial_obj, [False, False, True])
    if flip_vertical == True:
        axial_obj = sitk.Flip(axial_obj, [False, True, False])
    if long_axis == 'vertical':
        permutation_filter.SetOrder((0, 2, 1))
    if long_axis == 'horizontal':
        permutation_filter.SetOrder((2, 0, 1))
    coronal_obj = permutation_filter.Execute(axial_obj)

    return coronal_obj


def axial_to_saggital(sitk_obj, long_axis, flip_vertical):
    '''
    Function that reslices MRI scans with axial slices to saggital slices
    Parameters
    ----------
    sitk_obj: array like, sitk object (_, _, _)
        3D SimpleITK object, must have been read from a .nii(.gz) file.
        Must have axial slices.
    long_axis: string, choices = ['horizontal', 'vertical']
        Slice-wise axis to which the long edge of the brain is aligned
    flip_vertical: bool
        Whether to flip the input axial image vertically
    Outputs
    ----------
    axial_obj: array_like, sitk object (_, _, _)
        3D SimpleITK object, resliced to saggital plane
    '''
    permutation_filter = sitk.PermuteAxesImageFilter()
    axial_obj = sitk_obj
    axial_obj = sitk.Flip(axial_obj, [False, False, True])
    if flip_vertical == True:
        axial_obj = sitk.Flip(axial_obj, [False, True, False])
    if long_axis == 'vertical':
        permutation_filter.SetOrder((1, 2, 0))
    if long_axis == 'horizontal':
        permutation_filter.SetOrder((2, 1, 0))
    saggital_obj = permutation_filter.Execute(axial_obj)

    return saggital_obj


def getLargestConnectedComponent(data):
    '''
    Function that extracts the largest connected component from binary data
    Parameters
    ----------
    data: array like (_, _, _)
        3D binary data, corresponds to an image mask
    Outputs
    ----------
    data: array like (_, _, _)
        3D binary data, includes only the largest connected component
    '''
    c, n = snd.label(data)
    sizes = snd.sum(data, c, range(n+1))
    mask_size = sizes < (max(sizes))
    remove_voxels = mask_size[c]
    c[remove_voxels] = 0
    c[np.where(c != 0)] = 1
    data[np.where(c == 0)] = 0

    return data


def get_mri_major_axis(sitk_obj):
    '''
    Function that determines the slice-wise axis along which the long axis of
    the brain is aligned in a 3D MRI image. A binary mask is created, then the
    largest connected component in each slice is extracted. The fraction of the
    image canvas occupied by that connected component is calculated along the
    vertical and horizontal image axes. If there exists a consensus axis, that
    is returned as an output. Otherwise, a ValueError is raised.
    Parameters
    ----------
    sitk_obj: array like, sitk object (_, _, _)
        3D SimpleITK object, must have been read from a .nii(.gz) file.
    Outputs
    ----------
    major_axis: string, choices = ['vertical', 'horizontal']
        Slice-wise axis along which the long edge of the brain is aligned
    '''
    binarization_filter = sitk.OtsuThresholdImageFilter()
    binarization_filter.SetInsideValue(0)
    binarization_filter.SetOutsideValue(1)
    mask = binarization_filter.Execute(sitk_obj)
    img_array = sitk.GetArrayFromImage(mask)
    slice_major_axis = []
    for j in range(img_array.shape[0]):
        img_array[j, :, :] = getLargestConnectedComponent(
            img_array[j, :, :])
        pixelID_rows, pixelID_cols = np.where(
            img_array[j, :, :] == 1)
        try:
            xmax = np.amax(pixelID_cols)
            xmin = np.amin(pixelID_cols)
            ymax = np.amax(pixelID_rows)
            ymin = np.amin(pixelID_rows)
            x_extent = xmax - xmin
            y_extent = ymax - ymin
            if x_extent > y_extent:
                slice_major_axis.append('horizontal')
            else:
                slice_major_axis.append('vertical')
        except:
            pass
    horizontal_slices = slice_major_axis.count('horizontal')
    vertical_slices = slice_major_axis.count('vertical')

    if np.abs(horizontal_slices - vertical_slices) < 5:
        raise ValueError("Determination of MRI image orientation inconclusive. \
        The orientation of the MRI image supplied must be manually specified. \
        If the brain's 'long axis' is oriented vertically on a by-slice basis, \
        set --long_axis to 'vertical', or 'horizontal' if the opposite is true. \
        If the input image is sliced axially, ensure that is specified using \
        the input argument --mri_plane")
    elif horizontal_slices > vertical_slices:
        major_axis = 'horizontal'
    else:
        major_axis = 'vertical'

    print('Determined major axis automatically: {}'.format(major_axis))

    return major_axis


def pre_orientation_adjust(source_fn,
                           mri_plane,
                           rotate_90_degrees,
                           vertical_flip,
                           major_axis):

    source_img = sitk.ReadImage(source_fn)

    if mri_plane == 'axial':
        if rotate_90_degrees is True:
            source_img = rotate_90(source_img)
        if vertical_flip is True:
            source_img = flip_vertically(source_img)
    elif mri_plane == 'saggital':
        if major_axis == 'auto':
            major_axis = get_mri_major_axis(source_img)
        source_img = saggital_to_axial(source_img,
                                       major_axis,
                                       vertical_flip)
    elif mri_plane == 'coronal':
        if major_axis == 'auto':
            major_axis = get_mri_major_axis(source_img)
        source_img = coronal_to_axial(source_img,
                                      major_axis,
                                      vertical_flip)

    sitk.WriteImage(source_img, source_fn)

    return major_axis


def post_orientation_adjust(mask_fn,
                            mri_plane,
                            rotate_90_degrees,
                            vertical_flip,
                            major_axis):

    mask_img = sitk.ReadImage(mask_fn)

    if mri_plane == 'axial':
        if vertical_flip is True:
            mask_img = flip_vertically(mask_img)
        if rotate_90_degrees is True:
            mask_img = rotate_90(mask_img)
    elif mri_plane == 'saggital':
        mask_img = axial_to_saggital(mask_img,
                                     major_axis,
                                     vertical_flip)
    elif mri_plane == 'coronal':
        mask_img = axial_to_coronal(mask_img,
                                    major_axis,
                                    vertical_flip)

    sitk.WriteImage(mask_img, mask_fn)
