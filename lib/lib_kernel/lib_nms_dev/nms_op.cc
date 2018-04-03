/*
@author: zeming li
@contact: zengarden2009@gmail.com
*/
#include <stdio.h>
#include <cfloat>
#include <iostream>


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


// .Input("nms_overlap_thresh: float")
// .Input("max_out: int32")
REGISTER_OP("NMS")
.Attr("T: {float, double}")
.Attr("nms_overlap_thresh: float")
.Attr("max_out: int")
.Input("boxes: T")
.Output("keep_out: int32")
.Output("num_keep_out: int32")
.Output("mask_dev: uint64");


template <typename Device, typename T>
class NMSOp : public OpKernel {
public:
	explicit NMSOp(OpKernelConstruction* context) : OpKernel(context) {}
	void Compute(OpKernelContext* context) override {}
private:
  float nms_overlap_thresh;
  int max_out;
};

void NMSForward(const float* box_ptr, const int num_box,
    unsigned long long * mask_ptr, 
    int* output_ptr, 
    int* output_num_ptr, 
    float iou_threshold, 
    int max_output,
    const Eigen::GpuDevice& d);

//    const Tensor* bottom_rois, const Tensor* argmax_data,
//    const Tensor* out_backprop, const float spatial_scale, const int batch_size,
//    const int num_rois, const int height,  const int width, const int channels,
//    const int pooled_height, const int pooled_width,
//    const int sample_height, const int sample_width,
//    const TensorShape& tensor_output_shape)

// static void NmsOpKernel(
//     OpKernelContext* context, const int num_box, const Tensor* boxes,
//     const float nms_overlap_thresh, const int max_out,
//     Tensor* keep_out, Tensor * mask_dev)
// {
//   if (!context->status().ok()) {
//     return;
//   }
//   std::cout << "dbg: 0" << std::endl; 
//   std::cout << mask_dev->flat<unsigned long long>().data() << std::endl;
//   int * haha = keep_out->flat<int>().data();
//   haha[0] = 1;
//   NMSForward(
//     num_box,
//     boxes->flat<float>().data(),
//     nms_overlap_thresh,
//     max_out,
//     keep_out->flat<int>().data(),
//     mask_dev->flat<unsigned long long>().data(),
//     context->eigen_device<Eigen::GpuDevice>());
// }

static void NmsOpKernel(
    OpKernelContext* context, const Tensor* boxes, const int num_box, 
    TensorShape& mask_keep_shape,
    TensorShape& keep_output_shape, TensorShape& num_keep_output_shape, 
    const float nms_overlap_thresh, const int max_out) {
  Tensor* keep_out = nullptr;
  Tensor* num_keep_out = nullptr;
  Tensor* mask_keep = nullptr;




  OP_REQUIRES_OK(context, context->allocate_output(0, keep_output_shape, &keep_out));

  OP_REQUIRES_OK(context, context->allocate_output(1, num_keep_output_shape, &num_keep_out));
  OP_REQUIRES_OK(context, context->allocate_output(2, mask_keep_shape, &mask_keep));

  if (!context->status().ok()) {
    return;
  }
  NMSForward(
    boxes->flat<float>().data(),
    num_box,
    mask_keep->flat<unsigned long long>().data(),
    keep_out->flat<int>().data(),
    num_keep_out->flat<int>().data(),
    nms_overlap_thresh,
    max_out,
    context->eigen_device<Eigen::GpuDevice>());
}


template <class T>
class NMSOp<Eigen::GpuDevice, T>: public OpKernel {
public:
    typedef Eigen::GpuDevice Device;
    explicit NMSOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                       context->GetAttr("nms_overlap_thresh", &nms_overlap_thresh));
        OP_REQUIRES_OK(context,
                       context->GetAttr("max_out", &max_out));
    }
    
    void Compute(OpKernelContext* context) override {
        // Get input tensor
        const Tensor& det = context->input(0);
        // const float nms_overlap_thresh = context->input(1);
        // const int max_out = context->input(2);
        int num_box = det.dim_size(0);
        int box_dim = det.dim_size(1);

        OP_REQUIRES(context, det.dims() == 2,
            errors::InvalidArgument("det must be 2-dimensional"));

        //create output tensor

        int dim_keep_out[1];
        dim_keep_out[0] = max_out;
        TensorShape keep_output_shape;
        TensorShapeUtils::MakeShape(dim_keep_out, 1, &keep_output_shape);

        int dim_mask_keep[1];
        dim_mask_keep[0] = num_box * DIVUP(num_box, sizeof(unsigned long long) * 8);
        TensorShape mask_keep_shape;
        TensorShapeUtils::MakeShape(dim_mask_keep, 1, &mask_keep_shape);

        int dim_num_keep_out[1];
        dim_num_keep_out[0] = 1;
        TensorShape num_keep_output_shape;
        TensorShapeUtils::MakeShape(dim_num_keep_out, 1, 
            &num_keep_output_shape);




        // std::cout<< "mask_keep type " << output_values[0] << std::endl;
        if (!context->status().ok()) {
            return;
        }
        NmsOpKernel(
            context, &det, num_box, mask_keep_shape,
            keep_output_shape, num_keep_output_shape,
            nms_overlap_thresh, max_out);
    }
private:
  float nms_overlap_thresh;
  int max_out;
};


//REGISTER_KERNEL_BUILDER(Name("NMSOp").Device(DEVICE_GPU), NMSOp);
REGISTER_KERNEL_BUILDER(Name("NMS").Device(DEVICE_GPU).TypeConstraint<float>("T"), NMSOp<Eigen::GpuDevice, float>);
// #endif
// REGISTER_KERNEL_BUILDER(Name("NMSOp").Device(DEVICE_CPU), NMSOp);
