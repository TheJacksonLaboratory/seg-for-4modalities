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

import glob
import os
import joblib
import shutil
import sys
import time
import pandas as pd
import SimpleITK as sitk
from pathlib import PurePath
from .corrections import y_axis_correction, z_axis_correction
from ..scripts.segmentation import brain_seg_prediction
from ..scripts.original_seg import brain_seg_prediction_original
from .utils import get_suffix, write_backup_image, listdir_nohidden
from .utils import image_slice_4d, clip_outliers
from .quality import quality_check


def segment_file_structure_workflow(opt,
                                    voxsize,
                                    pre_paras,
                                    keras_paras):

    quality_check = pd.DataFrame(columns=['filename', 'slice_index', 'notes'])
    input_path_obj = PurePath(opt.input)

    if opt.input_type == 'dataset':
        qc_log_path = str(opt.input + '/segmentation_log.txt')
        mouse_dirs = sorted(listdir_nohidden(opt.input))
        print('Working with the following dataset directory: ' + opt.input)
        print('It contains the following subdirectories \
              corresponding to individual mice: \n' +
              str(mouse_dirs))
        sys.stderr = open(qc_log_path, 'w')
        for mouse_dir in mouse_dirs:
            modality_dirs = sorted(listdir_nohidden(mouse_dir))
            print('For the mouse ' +
                  str(mouse_dir) +
                  ' I see the following modality folders: \n ' +
                  str(modality_dirs))
            for modality_dir in modality_dirs:
                source_fn = glob.glob(os.path.join(modality_dir, '*'))[0]
                print('Starting Inference on file: ' + source_fn)
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
                    opt.frame_location)
                quality_check = quality_check.append(quality_check_temp,
                                                     ignore_index=True)
        sys.stderr.close()
        sys.stderr = sys.__stderr__

    elif opt.input_type == 'directory':
        print('Working with the following directory: ' + opt.input)
        print('It contains the following data files: \n' +
              str(listdir_nohidden(opt.input)))
        source_files = listdir_nohidden(opt.input)
        qc_log_path = str(opt.input + '/segmentation_log.txt')
        sys.stderr = open(qc_log_path, 'w')
        for source_fn in source_files:
            print('Starting Inference on file: ' + source_fn)
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
                opt.frame_location)
            quality_check = quality_check.append(quality_check_temp,
                                                 ignore_index=True)
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
            opt.frame_location)
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
                           frame_location):

    inference_start_time = time.time()
    suffix = get_suffix(z_axis_correction_check, y_axis_correction_check)

    # Basic image preprocessing. Unmodified image saved: {source}_original.nii
    source_path_obj, original_fn = write_backup_image(source_fn)
    image_slice_4d(source_fn,
                   best_frame=segmentation_frame,
                   frame_location=frame_location)
    clip_outliers(source_fn, clip_threshold=20)

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
                          y_axis_mask)

        if z_axis_correction_check == 'True':
            print('Performing y-axis correction to z-axis corrected data')
            z_axis_n4b_fn = str(z_axis_path_obj.with_name(
                z_axis_path_obj.stem.split('.')[0] +
                '_n4b' +
                ''.join(z_axis_path_obj.suffixes)))
            y_axis_correction(z_axis_fn,
                              z_axis_n4b_fn,
                              y_axis_mask)

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
                target_size,
                frac_patch=frac_patch,
                frac_stride=frac_stride,
                likelihood_categorization=likelihood_categorization)

    shutil.copyfile(original_fn, source_fn)

    print('Completed Inference - Time: ' +
          str(time.time() - inference_start_time))

    quality_check_list = pd.DataFrame(columns=['filename',
                                               'slice_index',
                                               'notes_1',
                                               'notes_2'])
    if quality_checks is True:
        qc_start_time = time.time()
        print('Performing post-inference quality checks: ' + source_fn)

        source_array = sitk.GetArrayFromImage(sitk.ReadImage(source_fn))
        mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_fn))
        qc_classifier = joblib.load(
            './msUNET/predict/scripts/quality_check_22822.joblib')
        file_quality_check_df = quality_check(source_array,
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
