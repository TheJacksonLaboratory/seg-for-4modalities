import SimpleITK as sitk
import numpy as np
import math


def resample_img(
        imgobj,
        new_spacing=None,
        interpolator=sitk.sitkLinear,
        target_size=None,
        target_spacing=None):
    # Function that resamples an SimpleITK image object to either a given spacing or to some target dimension
    # inputs: image object from sitk, usually .nii
    # Resample image to given voxel size
    # Define the resample object
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(imgobj.GetDirection())
    resample.SetOutputOrigin(imgobj.GetOrigin())
    # If no new size is specified, determine the desired output size from
    # spacing
    if target_size is None:
        resample.SetOutputSpacing(new_spacing)
        orig_size = np.array(imgobj.GetSize(), dtype=np.int)
        orig_spacing = np.array(imgobj.GetSpacing())
        target_size = orig_size * (orig_spacing / new_spacing)
        target_size = np.ceil(target_size).astype(
            np.int)  # Image dimensions are in integers
        target_size = [int(s) for s in target_size]
        # print(target_size)
    if new_spacing is None:
        orig_img_dims = np.array(sitk.GetArrayFromImage(imgobj).shape)
        # print(orig_img_dims)
        # resample.SetOutputSpacing(imgobj.GetSpacing())
        resample.SetOutputSpacing(target_spacing)
        target_size[2] = orig_img_dims[0]
        target_size[1] = int(
            np.floor(
                (target_size[0] /
                 orig_img_dims[1]) *
                orig_img_dims[2]))
        print(target_size)
    # Execute the resampling
    resample.SetSize(np.array(target_size, dtype='int').tolist())
    resampled_imgobj = resample.Execute(imgobj)
    return resampled_imgobj


def min_max_normalization(img):
    # Function the normalizes a numpy array such that it is suitable for NN
    # input
    new_img = img.copy()
    new_img = new_img.astype(np.float32)
    min_val = np.min(new_img)
    max_val = np.max(new_img)
    new_img = (np.asarray(new_img).astype(
        np.float32) - min_val) / (max_val - min_val)
    return new_img

# Write a function that subdivides an image into various sub-images.
# The patches will certainly overlap, and that's okay


def image_patch(img, img_params, frac_patch=None, frac_stride=None):
    # A function that handles patching of images for input numpy arrays converted
    # from SimpleITK objects. Subdivides images into overlapping 'patches' of
    # either a given integer size and stride, or a fraction of the input dimensions
    # Pull in requisite variables from external parameters
    patch_dims = [0, 0, 0]
    label_dims = [0, 0, 0]
    strides = [0, 0, 0]
    if frac_patch is None:
        patch_dims = img_params.patch_dims
        label_dims = img_params.patch_label_dims
        strides = img_params.patch_strides
    if frac_patch is not None:
        img_shape = list(img.shape)
        strides[0] = img_params.patch_strides[0]
        strides[1] = int(np.floor(img_shape[1] * frac_stride))
        strides[2] = int(np.floor(img_shape[2] * frac_stride))
        patch_dims[0] = img_params.patch_dims[0]
        patch_dims[1] = int(np.floor(img_shape[1] * frac_patch))
        patch_dims[2] = int(np.floor(img_shape[2] * frac_patch))
        label_dims[0] = img_params.patch_dims[0]
        label_dims[1] = int(np.floor(img_shape[1] * frac_patch))
        label_dims[2] = int(np.floor(img_shape[2] * frac_patch))
        print(patch_dims)

    n_class = img_params.n_class

    # Initialize output variables
    length, col, row = img.shape
    categorical_map = np.zeros((n_class, length, col, row), dtype=np.uint8)
    likelihood_map = np.zeros((length, col, row), dtype=np.float32)
    counter_map = np.zeros((length, col, row), dtype=np.float32)
    length_step = int(patch_dims[0] / 2)
    patch_counter = 0
    directory_size = (
        math.floor(
            (length - patch_dims[0] + 1) / strides[0])) * math.floor(
        ((col - patch_dims[1] + 1) / strides[1]) + 1) * math.floor(
                ((row - patch_dims[2] + 1) / strides[2]) + 1)
    image_directory = np.empty(
        [directory_size, patch_dims[1], patch_dims[2], patch_dims[0]])
    # Iterate through possible patch locations, save each to a directory
    for i in range(0, length - patch_dims[0] + 1, strides[0]):
        for j in range(0, col - patch_dims[1] + 1, strides[1]):
            for k in range(0, row - patch_dims[2] + 1, strides[2]):
                cur_patch = img[i:i + patch_dims[0],
                                j:j + patch_dims[1],
                                k:k + patch_dims[2]][:].reshape([1,
                                                                 patch_dims[0],
                                                                 patch_dims[1],
                                                                 patch_dims[2]])
                # Assume that data format has channels last
                cur_patch = np.transpose(cur_patch, (0, 2, 3, 1))
                image_directory[patch_counter, :, :, :] = cur_patch
                patch_counter += 1
    return image_directory


def mask_patch(img, img_params):
    # A function that handles patching of images for input numpy arrays converted
    # from SimpleITK objects. Subdivides images into overlapping 'patches' of
    # either a given integer size and stride, or a fraction of the input dimensions
    # Pull in requisite variables from external parameters
    patch_dims = img_params.patch_dims
    label_dims = img_params.patch_label_dims
    strides = img_params.patch_strides
    n_class = img_params.n_class

    # Initialize output variables
    length, col, row = img.shape
    categorical_map = np.zeros((n_class, length, col, row), dtype=np.uint8)
    likelihood_map = np.zeros((length, col, row), dtype=np.float32)
    counter_map = np.zeros((length, col, row), dtype=np.float32)
    length_step = int(patch_dims[0] / 2)
    patch_counter = 0
    # Should be a +1 after strides[0] in the first dimension, but this works
    directory_size = (
        math.floor(
            (length - patch_dims[0] + 1) / strides[0])) * math.floor(
        ((col - patch_dims[1] + 1) / strides[1]) + 1) * math.floor(
                ((row - patch_dims[2] + 1) / strides[2]) + 1)
    mask_directory = np.empty(
        [directory_size, patch_dims[1], patch_dims[2], patch_dims[0]])
    # Iterate through possible patch locations, save each to a directory
    for i in range(0, length - patch_dims[0] + 1, strides[0]):
        for j in range(0, col - patch_dims[1] + 1, strides[1]):
            for k in range(0, row - patch_dims[2] + 1, strides[2]):
                cur_patch = img[i:i + patch_dims[0],
                                j:j + patch_dims[1],
                                k:k + patch_dims[2]][:].reshape([1,
                                                                 patch_dims[0],
                                                                 patch_dims[1],
                                                                 patch_dims[2]])
                # Assume that data format has channels last
                cur_patch = np.transpose(cur_patch, (0, 2, 3, 1))
                mask_directory[patch_counter, :, :, :] = cur_patch
                patch_counter += 1
    return mask_directory


def uniform_mask(mask_path, mask_filenames):
    # Function that attempts to cast manually annotated masks into a consistent format.
    # Goal is to have binary masks with values 0 and 1 with float32 data types to
    # match that of the input data
    for i in range(0, len(mask_filenames)):
        mask_image = sitk.ReadImage(mask_path + '/' + mask_filenames[i])
        mask_array = sitk.GetArrayFromImage(mask_image)
        if np.amax(mask_array) == 1:
            pass
        if np.amax(mask_array) == 255:
            mask_array[mask_array == 255] = 1
        mask_image_standard = sitk.GetImageFromArray(mask_array)
        mask_image_standard.CopyInformation(mask_image)
        mask_image_standard_32 = sitk.Cast(mask_image, sitk.sitkFloat32)
        sitk.WriteImage(
            mask_image_standard_32,
            mask_path + '/' + mask_filenames[i])
