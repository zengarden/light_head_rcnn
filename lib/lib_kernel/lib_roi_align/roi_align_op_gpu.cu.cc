#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "roi_align_op_gpu.h"
#define F_DEVPTR(ptr) ((float*)((ptr)->desc.dev_ptr))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;

// namespace tensorflow {
using namespace tensorflow;


__device__ static float ROIAlignGetCoeff(float dh, float dw){
     dw = dw > 0 ? dw : -dw;
     dh = dh > 0 ? dh : -dh;
     return (1.0f - dh) * (1.0f - dw);
}

/**
  * Implementation of the bilinear interpolation.
  */
__device__ static float ROIAlignGetInterpolating(const float* data, const float h, 
  const float w, const int height, const int width, const int channels){
    float retVal = 0.0f;
    int h1 = floorf(h);
    int w1 = floorf(w);
    bool overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    retVal += overflow? 0.0f : data[(h1 * width + w1) * channels] * ROIAlignGetCoeff(h - float(h1), w - float(w1));
    h1 = ceilf(h);
    w1 = floorf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    retVal += overflow? 0.0f : data[(h1 * width + w1) * channels] * ROIAlignGetCoeff(h - float(h1), w - float(w1));
    h1 = floorf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    retVal += overflow? 0.0f : data[(h1 * width + w1) * channels] * ROIAlignGetCoeff(h - float(h1), w - float(w1));
    h1 = ceilf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    retVal += overflow? 0.0f : data[(h1 * width + w1) * channels] * ROIAlignGetCoeff(h - float(h1), w - float(w1));
    return retVal;
}
/**
  * Get the derivative of the bilinear interpolation.
  */
__device__ static void ROIAlignDistributeDiff(float* diff, const float top_diff, 
  const float h, const float w, const int height, const int width, 
  const int channels){
    int h1 = floorf(h);
    int w1 = floorf(w);
    bool overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow){
      atomicAdd(diff + (h1 * width + w1) * channels, 
        top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)));
    }
    h1 = ceilf(h);
    w1 = floorf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow){
      atomicAdd(diff + (h1 * width + w1) * channels, 
        top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)));
    }
    h1 = floorf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow){
      atomicAdd(diff + (h1 * width + w1) * channels, 
        top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)));
    }
    h1 = ceilf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow){
      atomicAdd(diff + (h1 * width + w1) * channels, 
        top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)));
    }
}

template <typename Dtype>
__global__ void RoiAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int height, const int width, 
    const int channels, const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, ph, pw, c) is an element in the pooled output
    int n = index;
    int c = n % channels;
    n /= channels;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    float roi_start_w = (bottom_rois[1]) * spatial_scale;
    float roi_start_h = (bottom_rois[2]) * spatial_scale;
    float roi_end_w = (bottom_rois[3]) * spatial_scale;
    float roi_end_h = (bottom_rois[4]) * spatial_scale;

    float roi_width = max(roi_end_w - roi_start_w, ((float)0.0));
    float roi_height = max(roi_end_h - roi_start_h, ((float)0.0));
    float bin_size_h = static_cast<float>(roi_height)
                       / static_cast<float>(pooled_height);
    float bin_size_w = static_cast<float>(roi_width)
                       / static_cast<float>(pooled_width);

    // regularly sample from a sample_height*sample_width grid
    // bottom_data += (roi_batch_ind * channels + c) * height * width;
    bottom_data += roi_batch_ind * channels * height * width;
    float sample_h_rate = 1.0f / float(sample_height);
    float sample_w_rate = 1.0f / float(sample_width);
    float hcenter;
    float wcenter;

    float tmp = float(-1e20);
    float tmp2;
    int buf_value = -1;
    for (int h_iter = 0; h_iter < sample_height; ++h_iter){
        for (int w_iter = 0; w_iter < sample_width; ++w_iter){
            hcenter = roi_start_h + bin_size_h * (ph + sample_h_rate * (h_iter + 0.5f));
            wcenter = roi_start_w + bin_size_w * (pw + sample_w_rate * (w_iter + 0.5f));
            tmp2 = ROIAlignGetInterpolating(bottom_data + c, hcenter, wcenter, 
              height, width, channels);
            if (tmp2 > tmp){
                tmp = tmp2;
                buf_value = w_iter + h_iter * sample_width;
            }
        }
    }
    top_data[index] = tmp;
    argmax_data[index] = buf_value;
  }
}

bool RoiAlignForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, 
    const int height, const int width, const int channels, 
    const int pooled_height, const int pooled_width, 
    const int sample_height, const int sample_width,
    const float* bottom_rois, float* top_data, int* argmax_data, 
    const Eigen::GpuDevice& d) {
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  cudaError_t err;

  RoiAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, spatial_scale, height, width, channels, 
      pooled_height, pooled_width, sample_height, sample_width, 
      bottom_rois, top_data, argmax_data);
  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}


template <typename Dtype>
__global__ void RoiAlignBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int height, const int width, const int channels, 
    const int pooled_height, const int pooled_width, 
    const int sample_height, const int sample_width,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, h, w, c) coords in bottom 
    int n = index;
    int c = n % channels;
    n /= channels;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    float roi_start_w = (bottom_rois[1]) * spatial_scale;
    float roi_start_h = (bottom_rois[2]) * spatial_scale;
    float roi_end_w = (bottom_rois[3]) * spatial_scale;
    float roi_end_h = (bottom_rois[4]) * spatial_scale;

    float roi_width = max(roi_end_w - roi_start_w, (float)0);
    float roi_height = max(roi_end_h - roi_start_h, (float)0);
    float bin_size_h = static_cast<float>(roi_height)
                       / static_cast<float>(pooled_height);
    float bin_size_w = static_cast<float>(roi_width)
                       / static_cast<float>(pooled_width);

    bottom_diff += roi_batch_ind * channels * height * width;

    float sample_h_rate = 1.0f / float(sample_height);
    float sample_w_rate = 1.0f / float(sample_width);

    float tmp = top_diff[index];
    int buffer_value = argmax_data[index];
    int w_iter = buffer_value % sample_width;
    int h_iter = buffer_value / sample_width;
    float hcenter = roi_start_h + bin_size_h * (ph + sample_h_rate * (h_iter + 0.5f));
    float wcenter = roi_start_w + bin_size_w * (pw + sample_w_rate * (w_iter + 0.5f));
    ROIAlignDistributeDiff(bottom_diff + c, tmp, hcenter, 
      wcenter, height, width, channels);
  }
}


bool RoiAlignBackwardLaucher(const float* top_diff, const float spatial_scale, 
  const int batch_size, const int num_rois,
    const int height, const int width, const int channels, 
    const int pooled_height, const int pooled_width, 
    const int sample_height, const int sample_width,
    const float* bottom_rois, float* bottom_diff, 
    const int* argmax_data, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  const int bottom_count = batch_size * height * width * channels;
  cudaError_t err;

  cudaMemsetAsync(bottom_diff, 0, sizeof(float) * bottom_count, d.stream());
  //cudaMemsetAsync(F_DEVPTR(bottom_diff), 0, sizeof(float) * bottom_count, d.stream());

  RoiAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                      kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, argmax_data, num_rois, spatial_scale, 
      height, width, channels, pooled_height, pooled_width, 
      sample_height, sample_width, bottom_diff, bottom_rois);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

// }  // namespace tensorflow

#endif  // GOOGLE_CUDA
