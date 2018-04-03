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
#define DIVUP(m,n) (((m) - 1) / (n) + 1)

template<typename T>
static inline T get_aligned_power2(T val, T align) {
    auto d = val & (align - 1);
    val += (align - d) & (align - 1);
    return val;
}

// .Input("nms_overlap_thresh: float")
// .Input("max_out: int32")
REGISTER_OP("NMS")
.Attr("T: {float, double}")
.Attr("nms_overlap_thresh: float")
.Attr("max_out: int")
.Input("boxes: T")
.Output("keep_out: int32")
.Output("num_keep_out: int32")
.Output("overlap_mask: uint64")
.Output("rm_mask: uint64");


template <typename Device, typename T>
class NMSOp : public OpKernel {
public:
	explicit NMSOp(OpKernelConstruction* context) : OpKernel(context) {}
	void Compute(OpKernelContext* context) override {}
private:
  float nms_overlap_thresh;
  int max_out;
};

void launch_gen_mask(
        const int nr_boxes, const float nms_overlap_thresh,
        const float *dev_boxes,
        const int dev_mask_width,
        unsigned long long *dev_mask,
        const Eigen::GpuDevice& d);

void launch_gen_indices(
        int nr_boxes, int max_output, int overlap_mask_width,
        const unsigned long long *overlap_mask,
        unsigned long long *rm_mask,
        int *out_idx,
        int *out_size,
        const Eigen::GpuDevice& d);


static void NmsOpKernel(
        OpKernelContext* context,
        const Tensor* boxes, const int num_box,
        TensorShape& overlap_mask_shape,
        TensorShape& rm_mask_shape,
        TensorShape& keep_out_shape,
        TensorShape& num_keep_output_shape,
        const float iou_threshold,
        const int max_output) {
    // generate tensors

    Tensor* keep_out = nullptr;
    Tensor* num_keep_out = nullptr;
    Tensor* overlap_mask = nullptr;
    Tensor* rm_mask = nullptr;


    OP_REQUIRES_OK(context, context->allocate_output(0, keep_out_shape, &keep_out));
    OP_REQUIRES_OK(context, context->allocate_output(1, num_keep_output_shape, &num_keep_out));
    OP_REQUIRES_OK(context, context->allocate_output(2,
                   overlap_mask_shape, &overlap_mask));
    OP_REQUIRES_OK(context, context->allocate_output(3, rm_mask_shape, &rm_mask));
    //batch = 1
    for (int i = 0; i < 1; ++ i) {
        launch_gen_mask(
            num_box, iou_threshold, boxes->flat<float>().data(),
            DIVUP(num_box, 64),
            overlap_mask->flat<unsigned long long>().data(),
            context->eigen_device<Eigen::GpuDevice>());

        launch_gen_indices(
            num_box, max_output, DIVUP(num_box, 64),
            overlap_mask->flat<unsigned long long>().data(),
            rm_mask->flat<unsigned long long>().data(),
            keep_out->flat<int>().data(),
            num_keep_out->flat<int>().data(),
            context->eigen_device<Eigen::GpuDevice>());
    }
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
        const Tensor& det = context->input(0);
        int num_box = det.dim_size(0);
        int box_dim = det.dim_size(1);

        OP_REQUIRES(context, det.dims() == 2,
            errors::InvalidArgument("det must be 2-dimensional"));

        int dim_keep_out[1];
        dim_keep_out[0] = max_out;
        TensorShape keep_output_shape;
        TensorShapeUtils::MakeShape(dim_keep_out, 1, &keep_output_shape);

        int dim_num_keep_out[1];
        dim_num_keep_out[0] = 1;
        TensorShape num_keep_output_shape;
        TensorShapeUtils::MakeShape(dim_num_keep_out, 1,
            &num_keep_output_shape);

        int dim_overlap_mask[1];
        int overlap_mask_bytes = num_box * DIVUP(num_box, 64) * sizeof(unsigned long long);
        int overlap_mask_bytes_align = get_aligned_power2(
                overlap_mask_bytes, 512);

        dim_overlap_mask[0] = overlap_mask_bytes_align / sizeof(unsigned long long);
        TensorShape overlap_mask_shape;
        TensorShapeUtils::MakeShape(dim_overlap_mask, 1, &overlap_mask_shape);

        int dim_rm_mask[1];
        dim_rm_mask[0] = DIVUP(num_box, 64);
        TensorShape rm_mask_shape;
        TensorShapeUtils::MakeShape(dim_rm_mask, 1, &rm_mask_shape);

        if (!context->status().ok()) {
            return;
        }
        NmsOpKernel(
            context, &det, num_box, overlap_mask_shape, rm_mask_shape,
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
