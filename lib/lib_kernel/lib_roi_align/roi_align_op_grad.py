import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import roi_align_op

@ops.RegisterGradient("RoiAlign")
def _roi_align_grad(op, grad, _):
  """The gradients for `roi_align`.
  Args:
    op: The `roi_align` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_align` op.
  Returns:
    Gradients with respect to the input of `roi align`.
  """
  data = op.inputs[0]
  rois = op.inputs[1]
  argmax = op.outputs[1]
  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')
  sample_height = op.get_attr('sample_height')
  sample_width = op.get_attr('sample_width')
  spatial_scale = op.get_attr('spatial_scale')

  # compute gradient
  data_grad = roi_align_op.roi_align_grad(data, rois, argmax, grad, 
    pooled_height, pooled_width, sample_height, sample_width, spatial_scale)

  return [data_grad, None]  # List of one Tensor, since we have one input
