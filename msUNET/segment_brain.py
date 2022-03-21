# author: Zachary Frohock
'''
Main function that handles inference for segmentation tasks

See User Guide for Inference for detailed information about inputs/outputs
'''

import os
from predict.core.utils import str2bool
from predict.core.utils import input_logging, save_quality_check
from predict.core.workflows import segment_file_structure_workflow
import SimpleITK as sitk
import argparse
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sitk.ProcessObject_SetGlobalWarningDisplay(False)

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
    default='auto')

opt = parser.parse_args()
input_logging(opt, sys.argv)


# Initialize parameter classes
class KerasParas:
    def __init__(self):
        self.model_path = None
        self.outID = 0
        self.thd = 0.5
        self.img_format = 'channels_last'
        self.loss = None


class PreParas:
    def __init__(self):
        self.patch_dims = []
        self.patch_label_dims = []
        self.patch_strides = []
        self.n_class = ''


pre_paras = PreParas()
pre_paras.patch_dims = [1, opt.image_patch, opt.image_patch]
pre_paras.patch_label_dims = [1, opt.image_patch, opt.image_patch]
pre_paras.patch_strides = [1, opt.image_stride, opt.image_stride]
pre_paras.n_class = 2

keras_paras = KerasParas()
keras_paras.outID = 0
keras_paras.thd = opt.threshold
keras_paras.loss = 'dice_coef_loss'
keras_paras.img_format = opt.channel_location
keras_paras.model_path = './msUNET/predict/scripts/' + opt.model

voxsize = 0.1

quality_check_results = segment_file_structure_workflow(opt,
                                                        voxsize,
                                                        pre_paras,
                                                        keras_paras)

save_quality_check(quality_check_results,
                   opt.input_type,
                   opt.input)
