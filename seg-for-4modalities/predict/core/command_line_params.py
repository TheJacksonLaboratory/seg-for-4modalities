# Copyright The Jackson Laboratory, 2022
# authors: Zachary Frohock
'''
Implimentation of the class CommandLineParams
'''

import sys
import os
import argparse
import textwrap
from argparse import RawTextHelpFormatter
from datetime import datetime
from .utils import str2bool, input_logging


class CommandLineParams:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-it",
            "--input_type",
            type=str,
            required=True,
            choices=['dataset',
                     'directory',
                     'file'])
        parser.add_argument(
            "-i",
            "--input",
            help="Directory containing specified file structure",
            type=str,
            required=True)
        parser.add_argument(
            "-m",
            "--model",
            help="Filename of the model to be used in inference",
            type=str,
            default='model156_0.50.25_scan.hdf5')
        parser.add_argument(
            "-th",
            "--threshold",
            help="Score threshold for selecting brain region",
            type=float,
            default=0.3)
        parser.add_argument(
            "-cl",
            "--channel_location",
            help="Whether the channels are the first or last dimension",
            type=str,
            default='channels_last',
            choices=[
                'channels_first',
                 'channels_last'])
        parser.add_argument(
            "-zc",
            "--z_axis_correction",
            help="Whether to perform z axis correction",
            type=str,
            default='False',
            choices=[
                'True',
                 'False'])
        parser.add_argument(
            "-yc",
            "--y_axis_correction",
            help="Whether to perform y axis correction",
            type=str,
            default='False',
            choices=[
                'True',
                 'False'])
        parser.add_argument(
            "-ns",
            "--new_spacing",
            help="Multiplicative factor by which image dimensions are divided. \
                  For 3D images, 3 values. \
                  First two correspond to single-slice dimensions, \
                  third to between-slice dimension.",
            nargs="*",
            type=float,
            default=[
                0.08,
                0.08,
                 1])  # usually 0.08,0.08,0.76
        parser.add_argument(
            "-ip",
            "--image_patch",
            help="Dimensions of the image patches to use to perform inference. \
                  Best to use the patch size from model training",
            type=int,
            default=128)
        parser.add_argument(
            "-is",
            "--image_stride",
            help="Stride of the image patching method to use in inference. \
                  Best to use the stride corresponding to that used in training",
            type=int,
            default=16)
        parser.add_argument(
            "-qc",
            "--quality_checks",
            help="Perform pre/post inference quality checks. \
                  If true, will output a list of IDs + slices that have an issue",
            type=str2bool,
            default=True)
        parse.add_argument(
            "-qt",
            "--quality_threshold",
            help="Score threshold for flagging slices for manual review",
            type=float,
            default=0.93)
        parser.add_argument(
            "-se",
            "--qc_skip_edges",
            default=False,
            type=str2bool)
        parser.add_argument(
            "-nt",
            "--normalization_mode",
            help="Use either by_slice or by_image to normalize images pre-inference",
            type=str,
            default="by_img",
            choices=[
                'by_slice',
                 'by_img'])
        parser.add_argument(
            "-ym",
            "--y_axis_mask",
            help="To use otsu threshold to mask y-axis correction pixels",
            type=str2bool,
            default=True)
        parser.add_argument(
            "-cs",
            "--constant_size",
            default=False,
            required=False,
            type=str2bool)
        parser.add_argument(
            "-ts",
            "--target_size",
            default=[
                188,
                188,
                None],
            nargs="*",
            type=int,
            required=False)
        parser.add_argument(
            "-ufp",
            "--use_frac_patch",
            default=False,
            required=False,
            type=str2bool)
        parser.add_argument(
            "-fp",
            "--frac_patch",
            default=0.5,
            required=False,
            type=float)
        parser.add_argument(
            "-fs",
            "--frac_stride",
            default=0.125,
            required=False,
            type=float)
        parser.add_argument(
            "-lc",
            "--likelihood_categorization",
            default=True,
            type=str2bool)
        parser.add_argument(  # Deprecated debug option
            "-sp",
            "--skip_preprocessing",
            help="Skip all preprocessing steps  \
                  and use original segmentation algorithm",
            type=str2bool,
            default=False)
        parser.add_argument(
            "-sf",
            "--segmentation_frame",
            help="If the input image is 4D, this value selects \
                  the frame on which inference should be run",
            type=int,
            default=0)
        parser.add_argument(
            "-fl",
            "--frame_location",
            help="Whether frames are the first or last index \
                  in the shape of a scan",
            type=str,
            choices=['frame_first',
                     'frame_last'],
            default='frame_first')
        parser.add_argument(
            "-oo",
            "--output_orientation",
            help="Orientation to which output mask will be cast",
            type=str,
            default='auto',
            choices=['auto'])
        parser.add_argument(
            "-bhf",
            "--binary_hole_filling",
            help='Whether to fill small holes and remove \
            isolated points in final masks',
            type=str2bool,
            default=False)
        parser.add_argument(
            "-mp",
            "--mri_plane",
            help="MRI plane of input images",
            type=str,
            choices=['sagittal',
                     'axial',
                     'coronal'],
            default='axial')
        parser.add_argument(
            "-r90",
            "--rotate_90_degrees",
            help="Rotate input image 90 degrees",
            type=str2bool,
            default="False")
        parser.add_argument(
            "-fv",
            "--flip_vertically",
            help="Flip input image vertically",
            type=str2bool,
            default=False)
        parser.add_argument(
            "-la",
            "--long_axis",
            help='Slice wise axis along which brain dimension is largest',
            choices=['horizontal',
                     'vertical',
                     'auto'],
            default='auto')

        opt = parser.parse_args()
        input_logging(opt, sys.argv)

        self.input_type = opt.input_type
        self.input = opt.input
        self.model = opt.model
        self.threshold = opt.threshold
        self.channel_location = opt.channel_location
        self.z_axis_correction = opt.z_axis_correction
        self.y_axis_correction = opt.y_axis_correction
        self.new_spacing = opt.new_spacing
        self.image_patch = opt.image_patch
        self.image_stride = opt.image_stride
        self.quality_checks = opt.quality_checks
        self.qc_skip_edges = opt.qc_skip_edges
        self.normalization_mode = opt.normalization_mode
        self.y_axis_mask = opt.y_axis_mask
        self.constant_size = opt.constant_size
        self.target_size = opt.target_size
        self.use_frac_patch = opt.use_frac_patch
        self.frac_patch = opt.frac_patch
        self.frac_stride = opt.frac_stride
        self.likelihood_categorization = opt.likelihood_categorization
        self.skip_preprocessing = opt.skip_preprocessing
        self.segmentation_frame = opt.segmentation_frame
        self.frame_location = opt.frame_location
        self.output_orientation = opt.output_orientation
        self.binary_hole_filling = opt.binary_hole_filling
        self.mri_plane = opt.mri_plane
        self.rotate_90_degrees = opt.rotate_90_degrees
        self.flip_vertically = opt.flip_vertically
        self.long_axis = opt.long_axis
        self.quality_threhold = opt.quality_threshold

    def check_parameters(self):
        '''
        Check that provided parameters comply with expected values
        '''
        pass
