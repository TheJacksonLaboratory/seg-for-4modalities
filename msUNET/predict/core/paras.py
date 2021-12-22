"""Defined Class for Transferring Parameters"""


class KerasParas:
    def __init__(self):
        self.model_path = None
        self.outID = 0
        self.thd = 0.5
        self.img_format = 'channels_first'
        self.loss = None


class PreParas:
    def __init__(self):
        self.patch_dims = []
        self.patch_label_dims = []
        self.patch_strides = []
        self.n_class = ''
