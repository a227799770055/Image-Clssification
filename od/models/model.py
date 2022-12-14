# -*- coding: utf-8 -*-
from addict import Dict
from torch import nn
import math
import yaml
import torch
import sys
sys.path.append('/home/insign/Doc/insign/flexible-yolov5')
from od.models.modules.common import Conv
from od.models.backbone import build_backbone
from od.models.neck import build_neck
from od.models.head import build_head
from utils.torch_utils import initialize_weights, fuse_conv_and_bn, model_info
import sys 
sys.path.append('/home/insign/Doc/insign/Python_utils/torch2trt')
from torch2trt import torch2trt
from torch2trt import TRTModule

class Model(nn.Module):
    def __init__(self, model_config):
        """
        :param model_config:
        """

        super(Model, self).__init__()
        if type(model_config) is str:
            model_config = yaml.full_load(open(model_config, 'r'))
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        backbone_out = self.backbone.out_shape

        self.fpn = build_neck('FPN', **backbone_out)
        fpn_out = self.fpn.out_shape

        fpn_out['version'] = model_config.backbone.version
        self.pan = build_neck('PAN', **fpn_out)

        pan_out = self.pan.out_shape
        model_config.head['ch'] = pan_out
        self.detection = build_head('YOLOHead', **model_config.head) #'YOLOHead replace with YOLOHead_trt for trt conversion
        self.stride = self.detection.stride
        self._initialize_biases()

        initialize_weights(self)

    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.detection  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for module in [self.backbone, self.fpn, self.pan, self.detection]:
            for m in module.modules():
                if type(m) is Conv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def forward(self, x):
        out = self.backbone(x)
        out = self.fpn(out)
        out = self.pan(out)
        out = self.detection(list(out))
        return out


if __name__ == '__main__':
    import os
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda')
    x = torch.zeros(1, 3, 480, 480).float().to(device)

    model = Model(model_config='/home/insign/Doc/insign/flexible-yolov5/configs/model_morph.yaml').eval().to(device)

    y = model(x)
    from od.models.head.head_predict import translate_to_predict
    y = translate_to_predict(y)
    
    weight = '/home/insign/Doc/insign/flexible-yolov5/Polyp/AI_box_0706_toTRT/weights/best.pt'
    model_a = torch.load(weight)['model']
    model_a.eval()
    model_a_weight = model_a.state_dict()

    model_weight = model.state_dict()
    model_weight.update(model_a_weight)
    model.load_state_dict(model_weight)
    model_trt = torch2trt(model, [x], fp16_mode=True)
    torch.save(model_trt.state_dict(), 'MorphYolo_backbone.pth')
