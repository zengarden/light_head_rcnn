/* Copyright 2015 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("PSAlignPool")
    .Attr("T: {float}")
    .Attr("group_size: int")
    .Attr("sample_height: int")
    .Attr("sample_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Output("top_data: T")
    .Output("mapping_channel: int32")
    .Output("argmax_position: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle dims;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &dims));
      ::tensorflow::shape_inference::DimensionHandle channels;
      channels = c->Dim(dims, 3);

      ::tensorflow::shape_inference::ShapeHandle dims_rois;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &dims_rois));
      ::tensorflow::shape_inference::DimensionHandle num_rois;
      num_rois = c->Dim(dims_rois, 0);

      int64 group_size;
      ::tensorflow::shape_inference::DimensionHandle psout_chl;
      TF_RETURN_IF_ERROR(c->GetAttr("group_size", &group_size));
      c->Divide(channels, group_size * group_size, true, &psout_chl);
      ::tensorflow::shape_inference::ShapeHandle output_shape =\
         c->MakeShape({num_rois, group_size, group_size, psout_chl});
      c->set_output(0, output_shape);
      //c->set_output(1, output_shape);
      return ::tensorflow::Status::OK();
    });



REGISTER_OP("PSAlignPoolGrad")
    .Attr("T: {float}")
    .Attr("sample_height: int")
    .Attr("sample_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("mapping_channel: int32")
    .Input("argmax_position: int32")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class PSAlignPoolOp : public OpKernel {
 public:
  explicit PSAlignPoolOp(OpKernelConstruction* context) : OpKernel(context) {}
   void Compute(OpKernelContext* context) override {}
  private:
  int group_size_;
  int sample_height_;
  int sample_width_;
  int spatial_scale_;
};


bool PSAlignPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, 
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width, 
    const int sample_height, const int sample_width,
    const float* bottom_rois, const int output_dim, 
    const int group_size, float* top_data, 
    int* mapping_channel, int* argmax_position,
    const Eigen::GpuDevice& d);

static void PSAlignPoolingKernel(
    OpKernelContext* context, const Tensor* bottom_data, 
    const float spatial_scale, const int num_rois, const int channels,
    const int height, const int width, 
    const int pooled_height, const int pooled_width, 
    const int sample_height, const int sample_width,
    const Tensor* bottom_rois, const int output_dim, const int group_size, 
    const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  Tensor* mapping_channel = nullptr;
  Tensor* argmax_position = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &mapping_channel));
  OP_REQUIRES_OK(context, context->allocate_output(2, tensor_output_shape, &argmax_position));

  if (!context->status().ok()) {
    return;
  }

  PSAlignPoolForwardLauncher(
    bottom_data->flat<float>().data(), spatial_scale, num_rois, channels,
    height, width, pooled_height, pooled_width, sample_height, sample_width,
    bottom_rois->flat<float>().data(), output_dim, group_size,
    output->flat<float>().data(), 
    mapping_channel->flat<int>().data(), 
    argmax_position->flat<int>().data(),
    context->eigen_device<Eigen::GpuDevice>());
}


template <class T>
class PSAlignPoolOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit PSAlignPoolOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("group_size", &group_size_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sample_height", &sample_height_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sample_width", &sample_width_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    int pooled_height = group_size_;
    
    int pooled_width = group_size_;
    int out_channels = num_channels / group_size_ / group_size_;
    
    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[3] = out_channels;
    dims[1] = pooled_height;
    dims[2] = pooled_width;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    PSAlignPoolingKernel(
      context, &bottom_data, spatial_scale_, num_rois, 
      num_channels, data_height, data_width, 
      pooled_height, pooled_width, sample_height_, sample_width_,
      &bottom_rois, out_channels, group_size_, output_shape);
  }
 private:
  int group_size_;
  int sample_height_;
  int sample_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("PSAlignPool").Device(DEVICE_GPU).TypeConstraint<float>("T"), PSAlignPoolOp<Eigen::GpuDevice, float>);

//align forward ok!


bool PSAlignPoolBackwardLauncher(
  const float* top_diff, 
  const int* mapping_channel, const int* argmax_position,
  const int batch_size,
  const int num_rois, const float spatial_scale, const int channels, 
  const int height, const int width, 
  const int pooled_height, const int pooled_width,
  const int sample_height, const int sample_width,
  const int output_dim, float* bottom_diff, 
  const float* bottom_rois, const Eigen::GpuDevice& d);

static void PSAlignPoolingGradKernel(
    OpKernelContext* context, 
    const Tensor* bottom_data, const Tensor* out_backprop, 
    const Tensor* mapping_channel, const Tensor* argmax_position, 
    const int batch_size, const int num_rois, 
    const float spatial_scale, const int channels, 
    const int height, const int width, 
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const int output_dim, const Tensor* bottom_rois, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  PSAlignPoolBackwardLauncher(
    out_backprop->flat<float>().data(), 
    mapping_channel->flat<int>().data(), 
    argmax_position->flat<int>().data(),
    batch_size, num_rois, spatial_scale, channels, height, width,
    pooled_height, pooled_width, sample_height, sample_width, output_dim, 
    output->flat<float>().data(), 
    bottom_rois->flat<float>().data(),
    context->eigen_device<Eigen::GpuDevice>());
}

// compute gradient
template <class Device, class T>
class PSAlignPoolGradOp : public OpKernel {
 public:
  explicit PSAlignPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("sample_height", &sample_height_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("sample_width", &sample_width_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& mapping_channel = context->input(2);
    const Tensor& argmax_position = context->input(3);
    const Tensor& out_backprop = context->input(4);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, mapping_channel.dims() == 4,
                errors::InvalidArgument("mapping_channel must be 4-dimensional"));

    OP_REQUIRES(context, argmax_position.dims() == 4,
                errors::InvalidArgument("argmax_position must be 4-dimensional"));
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int channels = bottom_data.dim_size(3);

    int pooled_height = out_backprop.dim_size(1);
    
    int pooled_width = out_backprop.dim_size(2);
    
    int output_dim = out_backprop.dim_size(3);
    
    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    PSAlignPoolingGradKernel(
      context, &bottom_data, &out_backprop, &mapping_channel, &argmax_position,
      batch_size, num_rois, spatial_scale_, channels, data_height, data_width, 
      pooled_height, pooled_width, sample_height_, sample_width_, 
      output_dim, &bottom_rois, output_shape);
  }
 private:
  int sample_height_;
  int sample_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("PSAlignPoolGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), PSAlignPoolGradOp<Eigen::GpuDevice, float>);
