import pickle
import tempfile
import zipfile
import shutil
import sys
import os
import SimpleITK as sitk
import numpy as np
from image import resample_img, image_patch, min_max_normalization


def layer_parse(max_trainable_layer, num_layers):
    # Deprecated
    # Function that produces a list of entries corresponding to the trainability
    # of convultional layers in the model
    layer_trainable = []
    for i in range(0, num_layers):
        if i < max_trainable_layer:
            layer_trainable.append(True)
        else:
            layer_trainable.append(False)
    return layer_trainable


def save_object(obj, filename):
    # Function that saves pickled objects
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def remove_x_y_data(zip_location, new_zip_location, file_ext_to_remove):
    # Function that works around an error in the Talos version required for this software.
    # Talos expects data to be 2D, but our image data is 3D. As such, it fails to
    # write representative x and y data to its experiment output directory created
    # via Talos.Deploy. As such, those files are missing. When it comes time to recreate
    # the Deploy object to save the Keras model, Talos.Resore fails. This function
    # writes some dummy 2D data where Talos expects it.
    zin = zipfile.ZipFile(zip_location, 'r')
    zout = zipfile.ZipFile(new_zip_location, 'w')
    for item in zin.infolist():
        buffer = zin.read(item.filename)
        if (item.filename[-6:] != file_ext_to_remove):
            zout.writestr(item, buffer)
    zout.close()
    zin.close()


def find_closest_power_2(x):
    # Deprecated
    # Function that finds the closests power of two larger than input value
    return 1 << (x - 1).bit_length()


def round_nearest_10(x):
    # Deprecated
    # Function that funds the nearest power of ten larger than input
    return int(round(x, -1))


def round_nearest_32(x):
    # Function that finds the nearest power of 32 larger than the input value
    # Used to determine a value appropriate for neural network input, allows
    # for padding of images to meet UNET input requirements
    return int(np.ceil(x / 32) * 32)


def pad_patches(data_list, mask_list, max_dim1, max_dim2):
    # Function that pads image patches with zeros such that they will be an appropriate size for NN input
    # If fractional patch size is specified, it is possible that the network will be trained on non-suqare patches
    # of arbitrary dimensions. As UNET architectures are relatively stringent with respect to input dimensions, those
    # arbitrary patch dimensions are padded with zeros to the nearest multiple of 32. This should prevent incompatibilities
    # between dimensions in expanding and contracting paths.
    max_dim1 = round_nearest_32(max_dim1)
    max_dim2 = round_nearest_32(max_dim2)
    #max_dim1 = max([max_dim1, max_dim2])
    #max_dim2 = max([max_dim1, max_dim2])
    max_dim = max([max_dim1, max_dim2])
    for i in range(len(data_list)):
        dir_dim1 = data_list[i][0].shape[0]
        dir_dim2 = data_list[i][0].shape[1]
        dim1_pad = max_dim1 - dir_dim1
        dim2_pad = max_dim2 - dir_dim2
        #dim1_pad = max_dim - dir_dim1
        #dim2_pad = max_dim - dir_dim2
        data_list[i] = np.pad(data_list[i], [[0, 0], [0, dim1_pad], [0, dim2_pad], [
                              0, 0]], mode='constant', constant_values=0)
        mask_list[i] = np.pad(mask_list[i], [[0, 0], [0, dim1_pad], [0, dim2_pad], [
                              0, 0]], mode='constant', constant_values=0)
    return data_list, mask_list, max_dim1, max_dim2


def load_data(
        data_path,
        data_filenames,
        mask_path,
        mask_filenames,
        interpolation_spacing,
        img_params,
        target_size=None,
        frac_patch=None,
        frac_stride=None):
    # Function that loads data from four modalities into a numpy array containing all data in the dataset directory
    # such that it is ready for neural netowrk input. Type coercion, image patching, outlier correction, etc. included.
    # Everything will be loaded into memeory. If training on a large dataset, ensure that hardware will accomodate.
    # Have to handle four modalities with different data structures. Do so here
    num_files = len(data_filenames)
    num_masks = len(mask_filenames)
    print(
        "Loading: " +
        str(num_files) +
        " data files and " +
        str(num_masks) +
        " masks")
    print('Begin loading data for patching')
    if num_masks != num_files:
        raise ValueError
    data_list = []
    mask_list = []
    modality_list = []
    if target_size is not None:
        spacing_list = []
        min_spacing = [None, None, None]
        for i in range(0, num_files):
            data_image = sitk.ReadImage(data_path + '/' + data_filenames[i])
            data_image_spacing = data_image.GetSpacing()
            spacing_list.append(data_image_spacing)
            min_spacing[0] = min([img_spacing[0]
                                 for img_spacing in spacing_list])
            min_spacing[1] = min([img_spacing[1]
                                 for img_spacing in spacing_list])
            min_spacing[2] = np.median(
                np.array([img_spacing[2] for img_spacing in spacing_list]))
    for i in range(0, num_files):
        current_modality = 'null'
        data_image = sitk.ReadImage(data_path + '/' + data_filenames[i])
        mask_image = sitk.ReadImage(mask_path + '/' + mask_filenames[i])

        # Determine which modality this file is
        mask_name = str(mask_filenames[i])
        # This is an ugly way to check
        if mask_name.find('anatomical') != -1:
            current_modality = 0
        elif mask_name.find('dti') != -1:
            current_modality = 1
        elif mask_name.find('noddi') != -1:
            current_modality = 2
        elif mask_name.find('regwarp') != -1:
            current_modality = 3
        elif mask_name.find('fmri') != -1:
            current_modality = 4

        # Print a message every 100 files loaded to keep it interesting
        if not i % 100:
            print(str(data_path +
                      '/' +
                      data_filenames[i]) +
                  ' and ' +
                  str(mask_path +
                      '/' +
                      mask_filenames[i]))
            print('We are loading the ' + str(i) + 'th image pair')
        data_image_spacing = data_image.GetSpacing()

        # Check for noddi images with an extra dimension. If it has one,
        # collapse it
        noddi_check_array = sitk.GetArrayFromImage(data_image)
        if len(noddi_check_array.shape) > 3:
            training_array = sitk.GetArrayFromImage(data_image)
            # We want the 8th time slice, it has the best image contrast
            noddi_check_array = noddi_check_array[7, :, :, :]
            training_img_noddi = sitk.GetImageFromArray(noddi_check_array)
            training_img_noddi.SetSpacing(data_image_spacing)
            data_image = training_img_noddi

        # Deal with masks
        if target_size is None:
            resampled_imgobj = resample_img(mask_image,
                                            new_spacing=interpolation_spacing,
                                            interpolator=sitk.sitkLinear)
        elif target_size is not None:
            resampled_imgobj = resample_img(mask_image,
                                            target_size=target_size,
                                            target_spacing=min_spacing,
                                            interpolator=sitk.sitkLinear)
        img_array = sitk.GetArrayFromImage(resampled_imgobj)
        np_mask_array = np.asarray(img_array)
        np_img_array = np.asarray(img_array)
        if np.amax(np_mask_array) == 1.0:
            super_threshold_indicies = np_mask_array > (0.6)
            sub_threshold_indicies = np_mask_array <= (0.6)
            np_mask_array[super_threshold_indicies] = 1
            np_mask_array[sub_threshold_indicies] = 0
        # For remaining modalities brain is labelled as 255
        elif np.amax(np_mask_array) == 255:
            super_threshold_indicies = np_mask_array > (255 / 1.8)
            sub_threshold_indicies = np_mask_array <= (255 / 1.8)
            np_mask_array[super_threshold_indicies] = 1
            np_mask_array[sub_threshold_indicies] = 0
        final_mask_array = np_mask_array
        cleaned_mask_array = final_mask_array

        # Deal with the data
        if target_size is None:
            resampled_imgobj = resample_img(data_image,
                                            new_spacing=interpolation_spacing,
                                            interpolator=sitk.sitkLinear)
        elif target_size is not None:
            resampled_imgobj = resample_img(data_image,
                                            target_size=target_size,
                                            target_spacing=min_spacing,
                                            interpolator=sitk.sitkLinear)
        img_array = sitk.GetArrayFromImage(resampled_imgobj)
        img_shape = img_array.shape
        img_array = img_array.flatten()

        # Clip data if it's above the 95th percentile
        clip_value = np.mean(img_array) * 20
        if frac_patch is None:
            replace_value = np.median(img_array)
        else:
            replace_value = np.mean(img_array) * 10
        img_array = np.where(img_array > clip_value, replace_value, img_array)
        img_array = np.reshape(img_array, img_shape)
        normed_array = min_max_normalization(img_array)
        final_data_array = normed_array
        cleaned_data_array = final_data_array

        # Eliminate the first and last layers of DTI, fMRI, regwarp and NODDI
        # images
        slice_to_delete = [0, -1]
        if current_modality != 0:
            cleaned_mask_array = np.delete(
                cleaned_mask_array, slice_to_delete, axis=0)
            cleaned_data_array = np.delete(
                cleaned_data_array, slice_to_delete, axis=0)

        # Eliminate image layers with a mask, but no relevant data
        # If an image has mask pixels, but has an average data value less than
        # 0.1, cut it output
        slice_to_delete = []
        for i in range(0, cleaned_data_array.shape[0]):
            current_data = cleaned_data_array[i, :, :]
            if not np.any(current_data[current_data > 0.2]):
                slice_to_delete.append(i)
        if slice_to_delete:
            cleaned_mask_array = np.delete(
                cleaned_mask_array, slice_to_delete, axis=0)
            cleaned_data_array = np.delete(
                cleaned_data_array, slice_to_delete, axis=0)

        # Build the dataset via patches
        if frac_patch is None:
            image_directory = image_patch(cleaned_data_array, img_params)
            mask_directory = image_patch(cleaned_mask_array, img_params)
        elif frac_patch is not None:
            image_directory = image_patch(
                cleaned_data_array, img_params, frac_patch, frac_stride)
            mask_directory = image_patch(
                cleaned_mask_array, img_params, frac_patch, frac_stride)
            shape_array = [x.shape for x in image_directory]
            # print(shape_array)

        # Build the modality list
        # Each entry corresponds to a single image patch. The dictionary will later be compared inside the model definition to
        # produce metrics at the end of a training cycle relating to each modality
        # The dictionary uses numerical keys relating to the five modalities. The keys can be found above
        # 0: anatomical, 1: dti, 2: noddi, 3: regwarp, 4: fmri
        current_modality_list = [current_modality] * image_directory.shape[0]

        # Append the current image to the overall directory
        data_list.append(image_directory)
        mask_list.append(mask_directory)
        modality_list.append(current_modality_list)

   # for i in range(0,len(data_list)):
   #     print(data_list[i].shape)
   #     print(data_filenames[i])
    print('Data loading complete')
    if frac_patch is None:
        return np.vstack(data_list), np.vstack(
            mask_list), np.concatenate(modality_list)
    elif frac_patch is not None:
        full_shape_array = [x.shape for x in data_list]
        #max_dim1 = max([patch.shape[0] for patch in image_dir for image_dir in data_list])
        max_dim1 = max([patch.shape[0]
                       for image_dir in data_list for patch in image_dir])
        #max_dim2 = max([patch.shape[1] for patch in image_dir for image_dir in data_list])
        max_dim2 = max([patch.shape[1]
                       for image_dir in data_list for patch in image_dir])
        data_list, mask_list, patch_dim1, patch_dim2 = pad_patches(
            data_list, mask_list, max_dim1, max_dim2)

        # for i in range(0,len(data_list)):
        # print(data_list[i].shape)
        # print(mask_list[i].shape)
        # print(data_filenames[i])

        return np.vstack(data_list), np.vstack(mask_list), np.concatenate(
            modality_list), patch_dim1, patch_dim2
