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
    TODO: Rewrite for robustness to different dimension orders

    Function that resamples input images for compatibility with inference model
    INPUTS
    imgobj: SimpleITK image object corresponding to raw data
    new_spacing: Muliplicative factor by which image dimensions should be
    multiplied by. Serves to change image size such that the patch dimension on
    which the network was trained fits reasonably into the input image. Of the
    form [horizontal_spacing, vertical_spacing, interlayer_spacing].
    If images are large, should have elements > 1
    If images are small should have elements < 1
    Interpolator: function used to interpolate in the instance
    new_spacing != [1,1,1]. Default is nearest neighbor as to not introduce
    new values.
    New_size: declared size for new images. If left as None will calculate
    the new size automatically
    baseon on new spacing and old image dimensions/spacing
    OUTPUTS
    resampled_imgobj: SimpleITK image object resampled in the desired manner
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
    '''
    return glob.glob(os.path.join(path, '*'))


def get_suffix(z_axis, y_axis):
    '''
    Function that determines the suffix to be used in finding the file in
    a modality folder on which final inference is to be run
    INPUTS
    z_axis: Was z-axis correction used, string 'True' if so
    y_axis: Was y_axis correction used, string 'True' if so
    OUTPUTS
    suffix: file suffix to be appended to source data filename for final
    inference
    '''
    suffix = ''
    if z_axis == 'True':
        suffix = '_z_axis'
    if y_axis == 'True':
        suffix = suffix + '_n4b'
    return suffix


def convex_hull_image(data):
    '''
    Function that calculates and draws the convex hull for a 2D binary image
    '''
    region = np.argwhere(data)
    hull = ConvexHull(region)
    verts = [(region[v, 0], region[v, 1]) for v in hull.vertices]
    img = Image.new('L', data.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T


def str2bool(v):
    '''
    A function that converts a collection of possible input values
    corresponding to True and False from strings too booleans
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
                       input_path):

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


def write_backup_image(source_fn):

    source_path_obj = PurePath(source_fn)
    original_fn = str(source_path_obj.with_name(
        source_path_obj.stem.split('.')[0] +
        '_original' +
        ''.join(source_path_obj.suffixes)))
    shutil.copyfile(source_fn, original_fn)

    return source_path_obj, original_fn


def image_slice_4d(source_fn,
                   best_frame,
                   frame_location):

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
        sitk.WriteImage(inference_img, source_fn)


def clip_outliers(source_fn, clip_threshold):

    source_image = sitk.ReadImage(source_fn)
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

    sitk.WriteImage(source_image, source_fn)


def remove_small_holes_and_points(img):

    binary_hole_filler = sitk.BinaryFillholeImageFilter()
    binary_inversion_filter = sitk.InvertIntensityImageFilter()
    binary_inversion_filter.SetMaximum(1)

    missing_hole_fill = binary_hole_filler.Execute(img)
    missing_hole_fill_invert = binary_inversion_filter.Execute(
        missing_hole_fill)
    isolated_brain_invert = binary_hole_filler.Execute(
        missing_hole_fill_invert)
    holes_filled_points_removed = binary_inversion_filter.Execute(
        isolated_brain_invert)

    return holes_filled_points_removed


def erode_img_by_slice(img,
                       kernel_radius):
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(kernel_radius)

    img_spacing = img.GetSpacing()
    img_array = sitk.GetArrayFromImage(img)

    eroded_array = np.zeros(shape=img_array.shape)
    for i in range(0, img_array.shape[0]):
        current_layer_img = sitk.GetImageFromArray(img_array[i, :, :])
        current_layer_img.SetSpacing(img_spacing)
        current_layer_img = erode_filter.Execute(current_layer_img)
        eroded_array[i, :, :] = sitk.GetArrayFromImage(current_layer_img)

    eroded_img = sitk.GetImageFromArray(eroded_array)
    eroded_img.SetSpacing(img_spacing)

    return eroded_array, eroded_img


def plot_intensity_comparison(intensity,
                              corrected_intensity,
                              filename):
    plt.figure()
    plt.plot(intensity)
    plt.plot(corrected_intensity)
    plt.xlabel('Slice Index')
    plt.ylabel('Mean Intensity in Preliminary Mask Region')
    plt.title('ROI Intensity - Z-Axis Correction')
    plt.legend(['Source Image', 'Z-Axis Corrected'])
    plt.savefig(filename)