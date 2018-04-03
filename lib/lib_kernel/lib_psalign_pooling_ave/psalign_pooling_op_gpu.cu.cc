#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "psalign_pooling_op_gpu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define CUDA_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;


using namespace tensorflow;

__device__ static float ROIAlignGetCoeff(float dh, float dw)
{
    dw = dw > 0 ? dw : -dw;
    dh = dh > 0 ? dh : -dh;
    return (1.0f - dh) * (1.0f - dw);
}

/**
  * Implementation of the bilinear interpolation.
  */
__device__ static float ROIAlignGetInterpolating(const float* data, const float h,
        const float w, const int height, const int width, const int channels)
{
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
        const int channels)
{
    int h1 = floorf(h);
    int w1 = floorf(w);
    bool overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow)
    {
        atomicAdd(diff + (h1 * width + w1) * channels,
                  top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)));
    }
    h1 = ceilf(h);
    w1 = floorf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow)
    {
        atomicAdd(diff + (h1 * width + w1) * channels,
                  top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)));
    }
    h1 = floorf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow)
    {
        atomicAdd(diff + (h1 * width + w1) * channels,
                  top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)));
    }
    h1 = ceilf(h);
    w1 = ceilf(w);
    overflow = (h1<0) || (w1<0) || (h1 >=height) || (w1>=width);
    if (!overflow)
    {
        atomicAdd(diff + (h1 * width + w1) * channels,
                  top_diff * ROIAlignGetCoeff(h - float(h1), w - float(w1)));
    }
}

template <typename Dtype>
__global__ void PSAlignPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel,
    int* argmax_position) {
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int n = index;
        int ctop = n % output_dim;
        n /= output_dim;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;

        /*
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;
        */

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];

        Dtype roi_start_w = static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
        Dtype roi_start_h = static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
        Dtype roi_end_w = static_cast<Dtype>(bottom_rois[3]) * spatial_scale;
        Dtype roi_end_h = static_cast<Dtype>(bottom_rois[4]) * spatial_scale;


        Dtype roi_width = max(roi_end_w - roi_start_w, 0.0);
        Dtype roi_height = max(roi_end_h - roi_start_h, 0.0);

        // Compute w and h at bottom
        Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

        //ps align max pooling
        /*
        bottom_data += roi_batch_ind * channels * height * width;
        float sample_h_rate = 1.0f / float(sample_height);
        float sample_w_rate = 1.0f / float(sample_width);
        float hcenter;
        float wcenter;
        int c = (ctop*group_size + ph)*group_size + pw;

        float tmp = float(-1e20);
        float tmp2;
        int buf_value = -1;
        for (int h_iter = 0; h_iter < sample_height; ++h_iter)
        {
            for (int w_iter = 0; w_iter < sample_width; ++w_iter)
            {
                hcenter = roi_start_h + bin_size_h * (ph + sample_h_rate * (h_iter + 0.5f));
                wcenter = roi_start_w + bin_size_w * (pw + sample_w_rate * (w_iter + 0.5f));
                tmp2 = ROIAlignGetInterpolating(
                  bottom_data + c, hcenter, wcenter, height, width, channels);
                if (tmp2 > tmp)
                {
                    tmp = tmp2;
                    buf_value = w_iter + h_iter * sample_width;
                }
            }
        }
        top_data[index] = tmp;
        argmax_position[index] = buf_value;
        mapping_channel[index] = c;
        */


        /* old ps ave align pooling*/
        int x1, x2, y1, y2;
        float px, py, pxmax, pymax, pxmin, pymin;
        pxmax = min(max(roi_start_w + static_cast<Dtype>(pw + 0.75) * bin_size_w, 0.001), width - 1.001);
        pymax = min(max(roi_start_h + static_cast<Dtype>(ph + 0.75) * bin_size_h, 0.001), height - 1.001);
        pxmin = min(max(roi_start_w + static_cast<Dtype>(pw + 0.25) * bin_size_w, 0.001), width - 1.001);
        pymin = min(max(roi_start_h + static_cast<Dtype>(ph + 0.25) * bin_size_h, 0.001), height - 1.001);

        Dtype out_sum = 0;
        int gw = pw;
        int gh = ph;
        int c = (ctop*group_size + gh)*group_size + gw;


        bottom_data += roi_batch_ind * channels * height * width;

        px = pxmin;
        py = pymin;

        x1 = floor(px);
        x2 = ceil(px);
        y1 = floor(py);
        y2 = ceil(py);

        out_sum += (px-x1)*(py-y1) * bottom_data[int(y2*width + x2)*channels+c];
        out_sum += (px-x1)*(y2-py) * bottom_data[int(y1*width + x2)*channels+c];
        out_sum += (x2-px)*(py-y1) * bottom_data[int(y2*width + x1)*channels+c];
        out_sum += (x2-px)*(y2-py) * bottom_data[int(y1*width + x1)*channels+c];

        px = pxmax;
        py = pymax;

        x1 = floor(px);
        x2 = ceil(px);
        y1 = floor(py);
        y2 = ceil(py);

        out_sum += (px-x1)*(py-y1) * bottom_data[int(y2*width + x2)*channels+c];
        out_sum += (px-x1)*(y2-py) * bottom_data[int(y1*width + x2)*channels+c];
        out_sum += (x2-px)*(py-y1) * bottom_data[int(y2*width + x1)*channels+c];
        out_sum += (x2-px)*(y2-py) * bottom_data[int(y1*width + x1)*channels+c];

        px = pxmin;
        py = pymax;

        x1 = floor(px);
        x2 = ceil(px);
        y1 = floor(py);
        y2 = ceil(py);

        out_sum += (px-x1)*(py-y1) * bottom_data[int(y2*width + x2)*channels+c];
        out_sum += (px-x1)*(y2-py) * bottom_data[int(y1*width + x2)*channels+c];
        out_sum += (x2-px)*(py-y1) * bottom_data[int(y2*width + x1)*channels+c];
        out_sum += (x2-px)*(y2-py) * bottom_data[int(y1*width + x1)*channels+c];

        px = pxmax;
        py = pymin;

        x1 = floor(px);
        x2 = ceil(px);
        y1 = floor(py);
        y2 = ceil(py);

        out_sum += (px-x1)*(py-y1) * bottom_data[int(y2*width + x2)*channels+c];
        out_sum += (px-x1)*(y2-py) * bottom_data[int(y1*width + x2)*channels+c];
        out_sum += (x2-px)*(py-y1) * bottom_data[int(y2*width + x1)*channels+c];
        out_sum += (x2-px)*(y2-py) * bottom_data[int(y1*width + x1)*channels+c];

        top_data[index] = out_sum / 4.0;
        mapping_channel[index] = c;

        // int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
        //                     + roi_start_h);
        // int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
        //                     + roi_start_w);
        // int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
        //                   + roi_start_h);
        // int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
        //                   + roi_start_w);
        // // Add roi offsets and clip to input boundaries
        // hstart = min(max(hstart, 0), height);
        // hend = min(max(hend, 0), height);
        // wstart = min(max(wstart, 0),width);
        // wend = min(max(wend, 0), width);
        // bool is_empty = (hend <= hstart) || (wend <= wstart);

        // int gw = pw;
        // int gh = ph;
        // int c = (ctop*group_size + gh)*group_size + gw;

        // //bottom_data += (roi_batch_ind * channels + c) * height * width;
        // bottom_data += roi_batch_ind * channels * height * width;
        // Dtype out_sum = 0;
        // for (int h = hstart; h < hend; ++h){
        //   for (int w = wstart; w < wend; ++w){
        //     //int bottom_index = h*width + w;
        //     int bottom_index = (h * width + w) * channels + c;
        //     out_sum += bottom_data[bottom_index];
        //   }
        // }

        // Dtype bin_area = (hend - hstart)*(wend - wstart);
        // top_data[index] = is_empty? 0. : out_sum / bin_area ;
        // mapping_channel[index] = c;
    }
}


bool PSAlignPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const float* bottom_rois, const int output_dim,
    const int group_size, float* top_data,
    int* mapping_channel, int* argmax_position,
    const Eigen::GpuDevice& d)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = num_rois * pooled_height * pooled_width * output_dim;
    cudaError_t err;

    PSAlignPoolingForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>(output_size, bottom_data, spatial_scale, channels, height, width,
            pooled_height, pooled_width, sample_height, sample_width,
            bottom_rois, output_dim, group_size,
            top_data, mapping_channel, argmax_position);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return d.ok();
}

template <typename Dtype>
__global__ void PSAlignPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int* argmax_position,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const int output_dim,
    Dtype* bottom_diff,
    const Dtype* bottom_rois)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        // The output is in order (n, ctop, ph, pw)

        /*
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int n = index / pooled_width / pooled_height / output_dim;
        */

        int n = index;
        n /= output_dim;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
        Dtype roi_start_w = static_cast<Dtype>(bottom_rois[1]) * spatial_scale;
        Dtype roi_start_h = static_cast<Dtype>(bottom_rois[2]) * spatial_scale;
        Dtype roi_end_w = static_cast<Dtype>(bottom_rois[3]) * spatial_scale;
        Dtype roi_end_h = static_cast<Dtype>(bottom_rois[4]) * spatial_scale;

        Dtype roi_width = max(roi_end_w - roi_start_w, (float)0);
        Dtype roi_height = max(roi_end_h - roi_start_h, (float)0);

        Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

        /*new roi align
        int c = mapping_channel[index];
        bottom_diff += roi_batch_ind * channels * height * width;

        float sample_h_rate = 1.0f / float(sample_height);
        float sample_w_rate = 1.0f / float(sample_width);

        float tmp = top_diff[index];
        int buffer_value = argmax_position[index];
        int w_iter = buffer_value % sample_width;
        int h_iter = buffer_value / sample_width;
        float hcenter = roi_start_h + bin_size_h * (ph + sample_h_rate * (h_iter + 0.5f));
        float wcenter = roi_start_w + bin_size_w * (pw + sample_w_rate * (w_iter + 0.5f));
        ROIAlignDistributeDiff(bottom_diff + c, tmp, hcenter,
                               wcenter, height, width, channels);
                               */

        
        /*old ps align */
        int x1, x2, y1, y2 ;
        float pxmin, pymin, pxmax, pymax, py, px;
        pxmax = min(max(roi_start_w + static_cast<Dtype>(pw + 0.75) * bin_size_w, 0.001), width - 1.001);
        pymax = min(max(roi_start_h + static_cast<Dtype>(ph + 0.75) * bin_size_h, 0.001), height - 1.001);
        pxmin = min(max(roi_start_w + static_cast<Dtype>(pw + 0.25) * bin_size_w, 0.001), width - 1.001);
        pymin = min(max(roi_start_h + static_cast<Dtype>(ph + 0.25) * bin_size_h, 0.001), height - 1.001);

        // Compute c at bottom
        int c = mapping_channel[index];
        Dtype* offset_bottom_diff = bottom_diff + roi_batch_ind * channels * height * width + c;
        Dtype diff_val = 0;
        diff_val = top_diff[index] / 4.0;

        px = pxmin;
        py = pymin;
        x1 = floor(px);
        x2 = ceil(px);
        y1 = floor(py);
        y2 = ceil(py);
        CudaAtomicAdd(offset_bottom_diff + int((y2*width + x2) * channels), diff_val * (px-x1)*(py-y1));
        CudaAtomicAdd(offset_bottom_diff + int((y1*width + x2) * channels), diff_val * (px-x1)*(y2-py));
        CudaAtomicAdd(offset_bottom_diff + int((y2*width + x1) * channels), diff_val * (x2-px)*(py-y1));
        CudaAtomicAdd(offset_bottom_diff + int((y1*width + x1) * channels), diff_val * (x2-px)*(y2-py));

        px = pxmax;
        py = pymax;
        x1 = floor(px);
        x2 = ceil(px);
        y1 = floor(py);
        y2 = ceil(py);
        CudaAtomicAdd(offset_bottom_diff + int((y2*width + x2) * channels), diff_val * (px-x1)*(py-y1));
        CudaAtomicAdd(offset_bottom_diff + int((y1*width + x2) * channels), diff_val * (px-x1)*(y2-py));
        CudaAtomicAdd(offset_bottom_diff + int((y2*width + x1) * channels), diff_val * (x2-px)*(py-y1));
        CudaAtomicAdd(offset_bottom_diff + int((y1*width + x1) * channels), diff_val * (x2-px)*(y2-py));

        px = pxmin;
        py = pymax;
        x1 = floor(px);
        x2 = ceil(px);
        y1 = floor(py);
        y2 = ceil(py);
        CudaAtomicAdd(offset_bottom_diff + int((y2*width + x2) * channels), diff_val * (px-x1)*(py-y1));
        CudaAtomicAdd(offset_bottom_diff + int((y1*width + x2) * channels), diff_val * (px-x1)*(y2-py));
        CudaAtomicAdd(offset_bottom_diff + int((y2*width + x1) * channels), diff_val * (x2-px)*(py-y1));
        CudaAtomicAdd(offset_bottom_diff + int((y1*width + x1) * channels), diff_val * (x2-px)*(y2-py));

        px = pxmax;
        py = pymin;
        x1 = floor(px);
        x2 = ceil(px);
        y1 = floor(py);
        y2 = ceil(py);
        CudaAtomicAdd(offset_bottom_diff + int((y2*width + x2) * channels), diff_val * (px-x1)*(py-y1));
        CudaAtomicAdd(offset_bottom_diff + int((y1*width + x2) * channels), diff_val * (px-x1)*(y2-py));
        CudaAtomicAdd(offset_bottom_diff + int((y2*width + x1) * channels), diff_val * (x2-px)*(py-y1));
        CudaAtomicAdd(offset_bottom_diff + int((y1*width + x1) * channels), diff_val * (x2-px)*(y2-py));
    }
}

bool PSAlignPoolBackwardLauncher(
    const float* top_diff,
    const int* mapping_channel, const int* argmax_position,
    const int batch_size,
    const int num_rois, const float spatial_scale, const int channels,
    const int height, const int width, 
    const int pooled_height, const int pooled_width, 
    const int sample_height, const int sample_width, 
    const int output_dim, float* bottom_diff,
    const float* bottom_rois, const Eigen::GpuDevice& d)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = num_rois * pooled_height * pooled_width * output_dim;
    const int bottom_size = batch_size * height * width * channels;
    cudaError_t err;

    // cudaMemsetAsync(bottom_diff, 0, sizeof(float) * bottom_count, d.stream());

    SetZero<<<(bottom_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
            kThreadsPerBlock, 0, d.stream()>>>(bottom_size, bottom_diff);

    PSAlignPoolingBackwardAtomic<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                                 kThreadsPerBlock, 0, d.stream()>>>(
                                     output_size, top_diff, mapping_channel, argmax_position,
                                     num_rois, spatial_scale, channels, height, width,
                                     pooled_height, pooled_width, sample_height, sample_width,
                                     output_dim, bottom_diff, bottom_rois);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return d.ok();
}

#endif  // GOOGLE_CUDA
