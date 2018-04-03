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
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;
// void NMSForward(const int num_box,
//                 const float* boxes,
//                 const float nms_overlap_thresh,
//                 const int max_out,
//                 int* keep_out,
//                 unsigned long long * mask_dev,
//                 const Eigen::GpuDevice& d);

// }
void NMSForward(const float* box_ptr, const int num_box,
	unsigned long long * mask_ptr, 
	int* output_ptr, 
	int* output_num_ptr, 
	float iou_threshold, 
	int max_output,
	const Eigen::GpuDevice& d);
}
// void NMSForward(const int num_box,
//                 const float* boxes,
//                 const float nms_overlap_thresh,
//                 const int max_out,
//                 int* keep_out, 
//                 unsigned long long * mask_dev,
//                 const Eigen::GpuDevice& d);
// }
#endif  // TENSORFLOW_CORE_KERNELS_NMS_OP_H_
