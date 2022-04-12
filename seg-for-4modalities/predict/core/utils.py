# author: Zachary Frohock
'''
Support functions for brain segmentation
'''

import glob
import os
import SimpleITK as sitk
import argparse
import shutil
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw
from pathlib import PurePath
from matplotlib import pyplot as plt


def min_max_normalization(img, normalization_mode='by_img'):
    '''
    Normalized input array, either by image or by slice.
    Parameters
    ----------
    img: array like (_, _, _)
    normalization_mode: String, either 'by_img' or 'by_slice'
        If by image, considers the maximum value over the entire stack.
        If by slice, considers max value over current slice only.
    Ouputs
    ----------
    new_img: array like (_, _, _)
        Normalized copy of input image
    '''
    new_img = img.copy()
    new_img = new_img.astype(np.float32)

    if normalization_mode == 'by_img':
        min_val = np.min(new_img)
        max_val = np.max(new_img)
        new_img = (np.asarray(new_img).astype(
            np.float32) - min_val) / (max_val - min_val)
    elif normalization_mode == 'by_slice':
        for slice_index in range(0, new_img.shape[0]):
            min_val = np.amin(new_img[slice_index, :, :])
            max_val = np.amax(new_img[slice_index, :, :])
            new_img[slice_index, :, :] = (new_img[slice_index, :, :].astype(
                np.float32) - min_val) / (max_val - min_val)
    else:
        raise ValueError("Normalization Mode is not correctly specified, \
                          use either 'by_img' or 'by_slice'")

    return new_img


def dim_2_categorical(label, num_class):
    '''
    Converts label values to integers, one per class
    Parameters
    ----------
    label: array like (_, _, _)
    num_class: int
        Number of classes into which segmentation splits pixels
    Ouputs
    ----------
    ex_label: array like (_, _, _)
        Label map with pixel values corresponding to class
    '''
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
    '''
    Function that resamples input images for compatibility with inference model.
    Assumes that the fist index corresponds to slices, remaining two indices describe
    the shape of each slice.
    Parameters:
        imgobj: array like (_, _, _)
        SimpleITK image object corresponding to raw data
    new_spacing: array-like, (_, _, _)
        Spacing to which an image will be resampled for inference. First two
        entries correspond to in-slice dimensions, third between slices.
    Interpolator: sitk interpolator class
        Function used to interpolate in the instancenew_spacing != [1,1,1].
        Default is nearest neighbor as to not introduce new values.
    target_size: array like (_, _, _) or None
        Dimensions to which all images will be sampled prior to patching.
        If None, images are not resampled to a constant size, and new_spacing
        is used to determine image resampling.
    revert: Bool
        If False, proceed with resampling as defined by either target_size or
        new_spacing as appropriate. If True, return resampled images to original
        dimensions for output.
    OUTPUTS
    resampled_imgobj: array like (_, _, _)
        SimpleITK image object resampled in the desired manner
    '''
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
    '''
    Function that lists the full paths of files and directories in path.
    Does not list hidden files.
    Parameters
    ----------
    path: string
        Path to directory
    Outputs
    ----------
        List of files in directory, less hidden files
    '''
    return glob.glob(os.path.join(path, '*'))


def listdir_only(path):
    '''
    Function that lists only subdirectories in the given path.
    Parameters
    ----------
    path: string
        Path to directory
    Outputs
    ----------
        List of subdirectories at path
    '''
    return [d for d in (os.path.join(path, d1) for d1 in os.listdir(path))
            if os.path.isdir(d)]


def list_nii_only(path):
    '''
    Function that lists only files with the .nii or .nii.gz extension.
    Parameters
    ----------
    path: string
        Path to directory
    Outputs
    ----------
        List of .nii or .nii.gz files at the path
    '''
    return [d for d in (os.path.join(path, d1) for d1 in os.listdir(path))
            if (d.endswith('.nii') or d.endswith('.nii.gz'))]


def get_suffix(z_axis, y_axis):
    '''
    Function that determines the suffix to be used in finding the file in
    a modality folder on which final inference is to be run
    Parameters
    ----------
    z_axis: String
        Was z-axis correction used, string 'True' if so
    y_axis: String
        Was y_axis correction used, string 'True' if so
    Outputs
    ----------
    suffix: String
        File suffix to be appended to source data filename for final
        inference
    '''
    suffix = ''
    if z_axis == 'True':
        suffix = '_z_axis'
    if y_axis == 'True':
        suffix = suffix + '_n4b'
    return suffix


def str2bool(v):
    '''
    A function that converts a collection of possible input values
    corresponding to True and False from strings too booleans
    Parameters
    ----------
    v: string
        User input
    Outputs
    ----------
    Bool corresponding to intended user input
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unable to coerce input to \
            boolean. Try 't/f', 'y/n', '1/0', etc.")


def input_logging(opt, input_command):
    '''
    Function that writes exact command passed to python to file, along with
    individual parameters
    Parameters
    ----------
    opt: Dictionary
        Input key-value pairs
    input_command: String
        Exact text passed in from command line
    Outputs
    ----------
    input_log
        Log of input command and parameters; written to disk
    '''
    if opt.input_type == 'dataset':
        input_log_path = str(opt.input + '/input_log.txt')
    elif opt.input_type == 'directory':
        input_log_path = str(opt.input + '/input_log.txt')
    elif opt.input_type == 'file':
        input_path_obj = PurePath(opt.input)
        input_log_path = str(input_path_obj.parents[0]) + '/input_log.txt'
    else:
        raise ValueError("Input type incorrectly specified, choose one of \
            'dataset', 'directory', or 'file'")

    with open(input_log_path, 'w') as input_log:
        input_log.write('Working Directory: ')
        input_log.write('\n')
        input_log.write(str(os.getcwd()))
        input_log.write('\n')
        input_log.write('\n')
        input_log.write('Command Passed to segment_brain.py: ')
        input_log.write('\n')
        input_log.write(" ".join(f"'{i}'" if " " in i
                        else i for i in input_command))
        input_log.write('\n')
        input_log.write('\n')
        input_log.write('Parameters Used: ')
        input_log.write('\n')
        for i in range(len(str(opt)[10:-1].split(','))):
            input_log.write(str(opt)[10:-1].split(',')[i].strip())
            input_log.write('\n')


def save_quality_check(quality_check,
                       input_type,
                       input_path,
                       do_quality_check):
    '''
    Function that saves quality check dataframe to disk
    Parameters
    ----------
    quality_check: Dataframe
        Structure containing information about slices in need of manual reivew
    input_type: String
        Type of input used, 'file', 'dataset', or 'directory'
    input_path:
        Path to input
    Outputs
    ----------
    quality_check; written to disk
    '''
    if len(quality_check) > 0:
        input_path_obj = PurePath(input_path)
        if input_type == 'file':
            print('Saving quality check file to: ' +
                  str(input_path_obj.parents[0]) +
                  '/quality_check.csv')
            quality_check.to_csv(str(input_path_obj.parents[0]) +
                                 '/quality_check.csv',
                                 index=False)
        else:
            print('Saving quality check file to: '
                  + input_path +
                  'quality_check.csv')
            quality_check.to_csv(input_path +
                                 '/quality_check.csv',
                                 index=False)
    elif do_quality_check == True:
        print('No slices in need of manual review - no quality_check.csv written.')


def write_backup_image(source_fn):
    '''
    Function that writes a copy of unmodified source data to disk
    Parameters
    ----------
    source_fn: String
        Path to unmodified source data
    Outputs
    ----------
    source_path_obj: Path obj
        Path to source as a PurePath object
    original_fn:
        Filename of source image copy
    '''
    source_path_obj = PurePath(source_fn)
    original_fn = str(source_path_obj.with_name(
        source_path_obj.stem.split('.')[0] +
        '_original' +
        ''.join(source_path_obj.suffixes)))
    shutil.copyfile(source_fn, original_fn)

    return source_path_obj, original_fn


def image_slice_4d(source_fn,
                   best_frame,
                   frame_location,
                   output_orientation):
    '''
    Function that slices imput 4d images to segment-able 3d images
    Parameters
    ----------
    source_fn: String
        Path to unmodified source data
    best_frame: Int
        Index of B0 frame in 4d stack. 0 indexed
    frame_location: String 'frame_first' or 'frame_last'
        For 4d input images, whether the index of the frame is first or last
    Outputs
    ----------
    inference_img
        3D slice of input 4D image; written to disk
    '''
    source_img = sitk.ReadImage(source_fn)
    source_spacing = source_img.GetSpacing()
    source_array = sitk.GetArrayFromImage(source_img)
    if len(source_array.shape) > 3:
        if frame_location == 'frame_first':
            inference_array = source_array[best_frame, :, :, :]
        else:
            inference_array = source_array[:, :, :, best_frame]
        inference_img = sitk.GetImageFromArray(inference_array)
        inference_img.SetSpacing(source_spacing)
        if source_img.GetPixelIDValue() != inference_img.GetPixelIDValue():
            inference_img = sitk.Cast(source_img,
                            source_img.GetPixelIDValue())
        sitk.WriteImage(inference_img,
                        source_fn)
    else:
        inference_img = source_img
        sitk.WriteImage(inference_img,
                        source_fn)

    return inference_img


def clip_outliers(source_fn,
                  clip_threshold,
                  output_orientation):
    '''
    Function that clips pixels with values much above the mean of the image
    Parameters
    ----------
    source_fn: String
        Path to unmodified source data
    clip_threshold: Int > 0
        Multiple of the mean above which pixel values will be clipped
    Outputs
    ----------
    clipped image; written to disk
    '''
    source_image = sitk.ReadImage(source_fn)
    original_dtype = source_image.GetPixelIDValue()
    source_spacing = source_image.GetSpacing()
    source_array = sitk.GetArrayFromImage(source_image)
    source_shape = source_array.shape

    clip_value = np.mean(source_array) * clip_threshold
    replace_value = np.median(source_array)

    source_array = np.where(
        source_array > clip_value,
        replace_value,
        source_array)

    source_array = np.reshape(source_array, source_shape)
    source_image = sitk.GetImageFromArray(source_array)
    source_image.SetSpacing(source_spacing)
    source_image = sitk.Cast(source_image, original_dtype)

    sitk.WriteImage(source_image,
                    source_fn)


def remove_small_holes_and_points(img):
    '''
    Function that removes isolated brain pixels and fills isolated non-
    brain pixels
    Parameters
    ----------
    img: array like (_, _, _)
        Sitk object corresponding to input image
    Output
    ----------
    holes_filled_points_removed: array like (_, _, _)
        Input image will holes filled and isolated points removed
    '''
    opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
    opening_filter.SetForegroundValue(1)
    opening_filter.SetKernelRadius(1)
    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
    closing_filter.SetForegroundValue(1)
    closing_filter.SetKernelRadius(1)
    holes_filled = closing_filter.Execute(img)
    holes_filled_points_removed = opening_filter.Execute(holes_filled)

    return holes_filled_points_removed


def erode_img_by_slice(img,
                       kernel_radius):
    '''
    Function that applies an erode filter to an input image by slice
    Parameters
    ----------
    img: array like (_, _, _)
        Sitk object corresponding to input image
    kernel_radius: Int > 0
        Radius of erosion kernel
    Outputs
    ----------
    eroded_array: array like (_, _, _)
        Numpy array of eroded input image
    eroded_img: array like (_, _, _)
        Sitk object of eroded input image
    '''
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(kernel_radius)
    erode_filter.SetForegroundValue(1)

    img_spacing = img.GetSpacing()
    img_array = sitk.GetArrayFromImage(img)

    eroded_array = np.zeros(shape=img_array.shape)
    for i in range(0, img_array.shape[0]):
        current_layer_img = sitk.GetImageFromArray(img_array[i, :, :])
        current_layer_img.SetSpacing(img_spacing)
        current_layer_img = erode_filter.Execute(sitk.Cast(current_layer_img,
                                                           sitk.sitkUInt8))
        eroded_array[i, :, :] = sitk.GetArrayFromImage(current_layer_img)

    eroded_img = sitk.GetImageFromArray(eroded_array)
    eroded_img.SetSpacing(img_spacing)

    return eroded_array, eroded_img


def plot_intensity_comparison(intensity,
                              corrected_intensity,
                              filename):
    '''
    Function that plots intensity of ROI before and after z-axis corrections
    Parameters
    ----------
    intensity: array like (_, )
        ROI intensity before corrections, elements individual slices
    corrected_intensity: array like (_, )
        ROI intensity after corrections. Elements individual slices
    Output
    ----------
    intensity by slice plot; written to disk
    '''
    plt.figure()
    plt.plot(intensity)
    plt.plot(corrected_intensity)
    plt.xlabel('Slice Index')
    plt.ylabel('Mean Intensity in Preliminary Mask Region')
    plt.title('ROI Intensity - Z-Axis Correction')
    plt.legend(['Source Image', 'Z-Axis Corrected'])
    plt.savefig(filename)
