# -*- coding: utf-8 -*-
from .yolo import YOLOHead, YOLOHead_trt

__all__ = ['build_head']
support_head = ['YOLOHead', 'YOLOHead_trt']


def build_head(head_name, **kwargs):
    assert head_name in support_head, f'all support head is {support_head}'
    head = eval(head_name)(**kwargs)
    return head