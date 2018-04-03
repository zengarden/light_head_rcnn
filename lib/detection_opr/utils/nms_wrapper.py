# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config import cfg
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms, cpu_soft_nms
import numpy as np

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):

    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep


def nms(dets, thresh, force_cpu=False):
  """Dispatch to either CPU or GPU NMS implementations."""

  if dets.shape[0] == 0:
    return []
  if not force_cpu:
    #return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    return gpu_nms(dets, thresh)
  else:
    return cpu_nms(dets, thresh)
