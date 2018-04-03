/*
@author: zeming li
@contact: zengarden2009@gmail.com
*/
#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_NMS_OP_H_
#define TENSORFLOW_USER_OPS_NMS_OP_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
//#define DIVUP(m,n) (((m) - 1) / (n) + 1)
// keepout and numout are the kernel output
/*
int* keep_out, int* num_out, 
*/
#define DIVUP(m,n) (((m) - 1) / (n) + 1)


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

}
#endif  // TENSORFLOW_CORE_KERNELS_NMS_OP_H_