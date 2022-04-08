# author: Zachary Frohock
'''
Functions that wrap segmentation workflow

segment_file_structure(...) handles the full segmentation workflow
                            from input file structure of a specified
                            type. See user guide for allowed file
                            structures

segment_image_workflow(...) handles segmentation workflow for one
                            single image in .nii format
'''

import os
import joblib
import shutil
import sys
import time
import traceback
import pathlib
import pandas as pd
import SimpleITK as sitk
from pathlib import PurePath
from .corrections import y_axis_correction, z_axis_correction
from ..scripts.segmentation import brain_seg_prediction
from ..scripts.original_seg import brain_seg_prediction_original
from .utils import get_suffix, write_backup_image
from .utils import image_slice_4d, clip_outliers, listdir_only, list_nii_only
from .orientation import get_orientation_string, pre_orientation_adjust
from .orientation import post_orientation_adjust, get_image_information
from .quality import quality_check


def segment_file_structure_workflow(opt,
                                    voxsize,
                                    pre_paras,
                                    keras_paras):
    '''
    Handles full segmentation workflow from an input file structure selected
    from the list of allowed structures. Curently 'file', 'directory', and
    'dataset'. See user guide for details on each file structure.
    Parameters
    ----------
    opt: Dictionary
        Contains input parameters specified in command line. Most relevant is
        opt.input, a string specifying file location and opt.input_type, a
        string specifying which input type to select.
    voxsize: Float
        Size of voxels in mm
    pre_paras: Class
        Class containing image processing parameters: patch dims, patch stride
    keras_paras: Class
        Class containing keras parameters for inference, including model path,
        threshold, and image format.
    Output
    ------
    quality_check: Dataframe
        Contains information about slices that are in need of manual review
        after inference.

    '''
    quality_check = pd.DataFrame(columns=['filename', 'slice_index', 'notes'])
    input_path_obj = PurePath(opt.input)

    if opt.input_type == 'dataset':
        qc_log_path = str(opt.input + '/segmentation_log.txt')
        sys.stderr = open(qc_log_path, 'w')

        mouse_dirs = sorted(listdir_only(opt.input))
        print('Working with the following dataset directory: ' + opt.input)
        print('It contains the following subdirectories '
              + 'corresponding to individual mice: \n'
              + str(mouse_dirs))

        for mouse_dir in mouse_dirs:
            modality_dirs = sorted(listdir_only(mouse_dir))
            print('For the mouse ' + str(mouse_dir) +
                  ' there exists the following modality directories: \n ' +
                  str(modality_dirs))

            for modality_dir in modality_dirs:
                try:
                    source_fns = list_nii_only(modality_dir)
                    if len(source_fns) == 0:
                        raise RuntimeError(
                            str('Zero files with extension .nii or .nii.gz'
                                + ' found in the dataset subdirectory '
                                + modality_dir
                                + ' Ensure that there is exactly 1 image file'
                                + ' in each subdirectory'))
                    source_fn = source_fns[0]
                    print('Starting Inference on file: ' + source_fn)
                    print('\nStarting Inference on file: ' + source_fn,
                          file=sys.stderr)
                    if len(source_fns) > 1:
                        raise RuntimeError(
                            str('Multiple files with extension .nii or .nii.gz'
                                + ' found in the dataset subdirectory '
                                + modality_dir
                                + ' Ensure that there is only 1 image file'
                                + ' in each subdirectory'))
                    quality_check_temp = segment_image_workflow(
                        source_fn,
                        opt.z_axis_correction,
                        opt.y_axis_correction,
                        voxsize,
                        pre_paras,
                        keras_paras,
                        opt.new_spacing,
                        opt.normalization_mode,
                        opt.constant_size,
                        opt.use_frac_patch,
                        opt.likelihood_categorization,
                        opt.y_axis_mask,
                        opt.frac_patch,
                        opt.frac_stride,
                        opt.quality_checks,
                        opt.qc_skip_edges,
                        opt.target_size,
                        opt.segmentation_frame,
                        opt.frame_location,
                        opt.output_orientation,
                        opt.binary_hole_filling,
                        opt.mri_plane,
                        opt.rotate_90_degrees,
                        opt.flip_vertically,
                        opt.long_axis)
                    quality_check = quality_check.append(quality_check_temp,
                                                         ignore_index=True)
                    print('Segmentation successful for the file: \n'
                          + str(source_fn),
                          file=sys.stderr)
                except Exception as e:
                    print('Segmentation has failed for the file: \n'
                          + str(source_fn)
                          + '\nWith the below error: \n',
                          file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    print('Segmentation failed for file: ' + source_fn)
                    print('See ' + str(qc_log_path) + ' for details \n')
                    continue

        sys.stderr.close()
        sys.stderr = sys.__stderr__

    elif opt.input_type == 'directory':
        qc_log_path = str(opt.input + '/segmentation_log.txt')
        sys.stderr = open(qc_log_path, 'w')

        print('Working with the following directory: ' + opt.input)
        print('It contains the following data files: \n' +
              str(list_nii_only(opt.input)))
        source_files = list_nii_only(opt.input)
        if len(source_files) == 0:
            raise RuntimeError(
                str('Zero files with extension .nii or .nii.gz'
                    + ' found in the dataset subdirectory '
                    + str(opt.input)
                    + ' Ensure that there is exactly 1 image file'
                    + ' in each subdirectory'))
        for source_fn in source_files:
            try:
                print('Starting Inference on file: ' + source_fn)
                print('\nStarting Inference on file: ' + source_fn,
                      file=sys.stderr)
                quality_check_temp = segment_image_workflow(
                    source_fn,
                    opt.z_axis_correction,
                    opt.y_axis_correction,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    opt.new_spacing,
                    opt.normalization_mode,
                    opt.constant_size,
                    opt.use_frac_patch,
                    opt.likelihood_categorization,
                    opt.y_axis_mask,
                    opt.frac_patch,
                    opt.frac_stride,
                    opt.quality_checks,
                    opt.qc_skip_edges,
                    opt.target_size,
                    opt.segmentation_frame,
                    opt.frame_location,
                    opt.output_orientation,
                    opt.binary_hole_filling,
                    opt.mri_plane,
                    opt.rotate_90_degrees,
                    opt.flip_vertically,
                    opt.long_axis)
                quality_check = quality_check.append(quality_check_temp,
                                                     ignore_index=True)
                print('Segmentation successful for the file: \n'
                      + str(source_fn),
                      file=sys.stderr)
            except Exception as e:
                print('Segmentation has failed for the file: \n'
                      + str(source_fn)
                      + '\nWith the below error: \n',
                      file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                print('Segmentation failed for file: ' + source_fn)
                print('See ' + str(qc_log_path) + ' for details \n')
                continue

        sys.stderr.close()
        sys.stderr = sys.__stderr__

    elif opt.input_type == 'file':
        if opt.skip_preprocessing is True:  # Debug option, currently disabled
            print('Skipping all preprocessing steps...')
            if opt.input is not None:
                output_filename = str(input_path_obj.with_name(
                    input_path_obj.stem.split('.')[0] +
                    '_mask' +
                    ''.join(input_path_obj.suffixes)))
                brain_seg_prediction_original(opt.input,
                                              output_filename,
                                              voxsize,
                                              pre_paras,
                                              keras_paras,
                                              opt.likelihood_categorization)
                exit()
        print('Performing inference on the following file: ' + str(opt.input))
        source_fn = opt.input
        print('Starting Inference on file: ' + source_fn)
        qc_log_path = str(input_path_obj.parents[0]) + '/segmentation_log.txt'
        sys.stderr = open(qc_log_path, 'w')
        try:
            quality_check_temp = segment_image_workflow(
                source_fn,
                opt.z_axis_correction,
                opt.y_axis_correction,
                voxsize,
                pre_paras,
                keras_paras,
                opt.new_spacing,
                opt.normalization_mode,
                opt.constant_size,
                opt.use_frac_patch,
                opt.likelihood_categorization,
                opt.y_axis_mask,
                opt.frac_patch,
                opt.frac_stride,
                opt.quality_checks,
                opt.qc_skip_edges,
                opt.target_size,
                opt.segmentation_frame,
                opt.frame_location,
                opt.output_orientation,
                opt.binary_hole_filling,
                opt.mri_plane,
                opt.rotate_90_degrees,
                opt.flip_vertically,
                opt.long_axis)
            print('Segmentation successful for the file: \n'
                  + str(source_fn),
                  file=sys.stderr)
        except Exception as e:
            print('Segmentation has failed for the file: \n'
                  + str(source_fn)
                  + '\nWith the below error: \n',
                  file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            print('Segmentation failed for file: ' + source_fn)
            print('See ' + str(qc_log_path) + ' for details \n')

        sys.stderr.close()
        sys.stderr = sys.__stderr__
        quality_check = quality_check.append(quality_check_temp,
                                             ignore_index=True)

    return quality_check


def segment_image_workflow(source_fn,
                           z_axis_correction_check,
                           y_axis_correction_check,
                           voxsize,
                           pre_paras,
                           keras_paras,
                           new_spacing,
                           normalization_mode,
                           constant_size,
                           use_frac_patch,
                           likelihood_categorization,
                           y_axis_mask,
                           frac_patch,
                           frac_stride,
                           quality_checks,
                           qc_skip_edges,
                           target_size,
                           segmentation_frame,
                           frame_location,
                           output_orientation,
                           binary_hole_filling,
                           mri_plane,
                           rotate_90_degrees,
                           flip_vertically,
                           major_axis):
    '''
    Controls workflow for segmentation of a single image stack. Includes
    both segmentation and preprocessing
    Parameters
    ----------
    source_fn: String
        Path to raw data file
    z_axis_correction_check: String
        Whether to perform z_axis correction before inference
    y_axis_correction_check: String
        Whether to perform y_axis correction before inference
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
    constant_size: Bool
        Whether all images should be sampled to a constant size before patching
    use_frac_patch: Bool
        Whether image patches should be user-defined constants or a fraction
        of image dimensions (True)
    likelihood_categorization: Bool
        How should final binarization of score -> mask be done. If True, use
        the max value of likelihood per-pixel. If False, use the mean value.
    y_axis_mask: Bool
        Whether to apply n4bias field correction to the entire image (False),
        or an binary otsu masked region (True). Only applies if
        y_axis_correction_check is True
    frac_patch: Float in range (0, 1)
        If use_frac_patch is True, this values determines the fraction of
        resampled image dimensions the patch size should be set to
    frac_stride: Float in range (0,1)
        If use_frac_patch is True, the fraction of resampled image dimensions
        the patch stride should be set to
    quality_checks: Bool
        If true, perform post-inference quality checks. If false, do not.
    qc_skip_edges: Bool
        If true, quality check processing will not consider first and last
        slices.
    target_size: array like (_, _, _)
        If constant_size is True, this is the dimensions to which all images
        will be sampled prior to patching
    segmentation_frame: Int
        For 4d input images, this is the index of the B0 frame. 0-indexed
    frame_location: String 'frame_first' or 'frame_last'
       For 4d input images, whether the index of the frame is first or last
    Output
    -----
    quality_check_list: Dataframe
       Contains information about slices that are in need of manual review
       after inference.
    backup_image:
        Copy of unmodified source image; written to disk
    '''
    inference_start_time = time.time()

    suffix = get_suffix(z_axis_correction_check, y_axis_correction_check)
    source_path_obj, original_fn = write_backup_image(source_fn)
    segmentation_fn = str(source_path_obj.with_name(
            source_path_obj.stem.split('.')[0] +
            '_segmentation' +
            ''.join(source_path_obj.suffixes)))

    input_orientation = get_orientation_string(sitk.ReadImage(source_fn))
    input_image_information = get_image_information(source_fn)

    if output_orientation == 'auto':
        output_orientation = input_orientation

    inference_img = image_slice_4d(source_fn,
                                   best_frame=segmentation_frame,
                                   frame_location=frame_location,
                                   output_orientation=output_orientation)
    clip_outliers(source_fn,
                  clip_threshold=20,
                  output_orientation=output_orientation)

    input_major_axis = pre_orientation_adjust(source_fn,
                                              mri_plane,
                                              rotate_90_degrees,
                                              flip_vertically,
                                              major_axis)

    if z_axis_correction_check == 'True':
        print('Performing z-axis correction')
        z_axis_fn = str(source_path_obj.with_name(
            source_path_obj.stem.split('.')[0] +
            '_z_axis' +
            ''.join(source_path_obj.suffixes)))
        z_axis_path_obj = PurePath(z_axis_fn)
        if not use_frac_patch:
            if not constant_size:
                z_axis_correction(
                    source_fn,
                    z_axis_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    new_spacing,
                    normalization_mode,
                    output_orientation,
                    binary_hole_filling,
                    likelihood_categorization=likelihood_categorization)
            elif constant_size:
                z_axis_correction(
                    source_fn,
                    z_axis_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    new_spacing,
                    normalization_mode,
                    output_orientation,
                    binary_hole_filling,
                    target_size,
                    likelihood_categorization=likelihood_categorization)
        elif use_frac_patch:
            if not constant_size:
                z_axis_correction(
                    source_fn,
                    z_axis_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    new_spacing,
                    normalization_mode,
                    output_orientation,
                    binary_hole_filling,
                    frac_patch=frac_patch,
                    frac_stride=frac_stride,
                    likelihood_categorization=likelihood_categorization)
            elif constant_size:
                z_axis_correction(
                    source_fn,
                    z_axis_fn,
                    voxsize,
                    pre_paras,
                    keras_paras,
                    new_spacing,
                    normalization_mode,
                    output_orientation,
                    binary_hole_filling,
                    target_size,
                    frac_patch=frac_patch,
                    frac_stride=frac_stride,
                    likelihood_categorization=likelihood_categorization)

    if y_axis_correction_check == 'True':
        print('Performing y-axis correction to source data')
        y_axis_fn = str(source_path_obj.with_name(
            source_path_obj.stem.split('.')[0] +
            '_n4b' +
            ''.join(source_path_obj.suffixes)))
        y_axis_correction(source_fn,
                          y_axis_fn,
                          y_axis_mask,
                          output_orientation)

        if z_axis_correction_check == 'True':
            print('Performing y-axis correction to z-axis corrected data')
            z_axis_n4b_fn = str(z_axis_path_obj.with_name(
                z_axis_path_obj.stem.split('.')[0] +
                '_n4b' +
                ''.join(z_axis_path_obj.suffixes)))
            y_axis_correction(z_axis_fn,
                              z_axis_n4b_fn,
                              y_axis_mask,
                              output_orientation)

    final_inference_fn = str(source_path_obj.with_name(
        source_path_obj.stem.split('.')[0] +
        suffix +
        ''.join(source_path_obj.suffixes)))
    mask_fn = str(source_path_obj.with_name(
        source_path_obj.stem.split('.')[0] +
        '_mask' +
        ''.join(source_path_obj.suffixes)))
    if not use_frac_patch:
        if not constant_size:
            brain_seg_prediction(
                final_inference_fn,
                mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                output_orientation,
                binary_hole_filling,
                likelihood_categorization=likelihood_categorization)
        elif constant_size:
            new_spacing = None
            brain_seg_prediction(
                final_inference_fn,
                mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                output_orientation,
                binary_hole_filling,
                target_size,
                likelihood_categorization=likelihood_categorization)
    if use_frac_patch:
        if not constant_size:
            brain_seg_prediction(
                final_inference_fn,
                mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                output_orientation,
                binary_hole_filling,
                frac_patch=frac_patch,
                frac_stride=frac_stride,
                likelihood_categorization=likelihood_categorization)
        elif constant_size:
            new_spacing = None
            brain_seg_prediction(
                final_inference_fn,
                mask_fn,
                voxsize,
                pre_paras,
                keras_paras,
                new_spacing,
                normalization_mode,
                output_orientation,
                binary_hole_filling,
                target_size,
                frac_patch=frac_patch,
                frac_stride=frac_stride,
                likelihood_categorization=likelihood_categorization)

    post_orientation_adjust(mask_fn,
                            mri_plane,
                            rotate_90_degrees,
                            flip_vertically,
                            input_major_axis,
                            input_image_information)
    post_orientation_adjust(mask_fn.split('.nii')[0] + '_likelihood.nii',
                            mri_plane,
                            rotate_90_degrees,
                            flip_vertically,
                            input_major_axis,
                            input_image_information)

    shutil.copyfile(source_fn, segmentation_fn)
    shutil.copyfile(original_fn, source_fn)
    os.remove(original_fn)

    print('Completed Inference - Time: ' +
          str(time.time() - inference_start_time))

    quality_check_list = pd.DataFrame(columns=['filename',
                                               'slice_index',
                                               'notes_1',
                                               'notes_2'])
    if quality_checks is True:
        qc_start_time = time.time()
        print('Performing post-inference quality checks: ' + source_fn)

        inference_array = sitk.GetArrayFromImage(inference_img)
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_fn))
        qc_classifier = joblib.load(
            str(pathlib.Path(__file__).parent.resolve()).split('core')[0]
            + 'scripts/quality_check_22822.joblib')  # Replace with more robust
        file_quality_check_df = quality_check(inference_array,
                                              mask_array,
                                              qc_classifier,
                                              source_fn,
                                              mask_fn,
                                              qc_skip_edges)
        quality_check_list = quality_check_list.append(file_quality_check_df,
                                                       ignore_index=True)

        print('Completed Quality Checks - Time: ' +
              str(time.time() - qc_start_time))

    return quality_check_list
