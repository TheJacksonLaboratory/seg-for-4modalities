# author: Zachary Frohock
'''
Main function that handles inference for segmentation tasks

See User Guide for Inference for detailed information about inputs/outputs
'''

import os
import sys
import pathlib
from .predict.core.utils import save_quality_check
from .predict.core.command_line_params import CommandLineParams
from .predict.core.workflows import segment_file_structure_workflow
import SimpleITK as sitk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sitk.ProcessObject_SetGlobalWarningDisplay(False)


def main():
    opt = CommandLineParams()
    opt.check_parameters()

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
    keras_paras.model_path = str(str(pathlib.Path(__file__).parent.resolve())
                             + str('/predict/scripts/')
                             + str(opt.model))

    voxsize = 0.1

    quality_check_results = segment_file_structure_workflow(opt,
                                                            voxsize,
                                                            pre_paras,
                                                            keras_paras)

    save_quality_check(quality_check_results,
                       opt.input_type,
                       opt.input)

    sys.exit(0)


if __name__ == '__main__':
    main()
