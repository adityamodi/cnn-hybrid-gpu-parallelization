#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#include "readubyte.h"

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define NUM_GPUS 4

// addresses (rank) for each worker
#define STEP_SIZE 0.0001
#define BLOCK_WIDTH 128
#define ITERATIONS 3350
#define NUM_WORKERS 2
#define NUM_WORKERS_DATA_PARALLELISM 2
#define BATCH_SIZE 32

#define TRAIN_IMAGES "train-images-idx3-ubyte"
#define TRAIN_LABELS "train-labels-idx1-ubyte"
#define TEST_IMAGES "t10k-images-idx3-ubyte"
#define TEST_LABELS "t10k-labels-idx1-ubyte"

#define CONV1_FILTERS 20
#define CONV1_FILTER_SIZE 5
#define CONV2_FILTERS 50
#define CONV2_FILTER_SIZE 5
#define POOL_SIZE 2
#define POOL_STRIDE 2

#define IMG_SIZE 28
#define CONV1_OUTPUT_DIM (IMG_SIZE - CONV1_FILTER_SIZE + 1) / POOL_STRIDE
#define CONV2_OUTPUT_DIM (CONV1_OUTPUT_DIM - CONV2_FILTER_SIZE + 1) / POOL_STRIDE

#define FORPROP_TAG 0
#define BACKPROP_TAG 1
#define FORPROP_LABELS_TAG 2
#define SYNC_CONV1_TAG 3
#define SYNC_CONV1BIAS_TAG 4
#define SYNC_CONV2_TAG 5
#define SYNC_CONV2BIAS_TAG 6
#define BACKPROP_FC_TAG 7
#define TRAINING_LOSS_TAG 8
#define SEND_PREDICTIONS_TAG 9
#define FLAG_TAG 2001
#define FORPROP_FC1_TAG 15

#define RANDOM_SEED -1

#define CONV_ADDRESS_1 0
#define CONV_ADDRESS_2 1
#define FC_ADDRESS_1 2
#define FC_ADDRESS_2 3

#define FC1_OUTPUT_DIM 240
// #define FC1_OUTPUT_DIM_FRACTION 200
#define FC2_OUTPUT_DIM 10
// #define FC2_OUTPUT_DIM_FRACTION 5

// #define TEST_SET_SIZE 10000
#define TEST_SET_SIZE 10000
#define TEST_INTERVAL 100

__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    vec[idx] = 1.0f;
}

__global__ void Correct_order(float *recv, float *corr, int size, int batch_size, int fc_size, int num_workers){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= size) return;
    int len = batch_size * fc_size;
    int worker_id = (idx / len);
    int img_id = (idx % len) / fc_size;
    int act_id = worker_id * fc_size + (idx % len) % fc_size;
    corr[img_id * fc_size * num_workers + act_id] = recv[idx];
    return;
}

__global__ void der_correct_order(float *recv, float *corr, int size, int batch_size, int fc_size, int num_workers, int my_id){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx >= size) return;
    int FC_size = fc_size * num_workers;
    int len = batch_size * FC_size;
    int worker_id = (idx / len);
    int img_id = (idx % len) / FC_size;
    int act_id = (idx % len) % FC_size;
    if (act_id < fc_size * my_id || act_id >= fc_size * (my_id + 1)) return;
    corr[worker_id * fc_size * batch_size + img_id * fc_size + act_id - fc_size * my_id] = recv[idx];
    return;
}

//<<<RoundUp(softmax_context.m_batchsize, BLOCK_WIDTH), BLOCK_WIDTH>>>
__global__ void MergeMatrices(float *d_fc1relu_1, float *d_fc1relu_2, float *d_fc1relu, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size)
        return;
    d_fc1relu[idx] = d_fc1relu_1[idx];
    d_fc1relu[idx + size] = d_fc1relu_2[idx];
}

__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

__global__ void copyData(float *data1, float *data2, int len){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < len) data2[idx] = data1[idx];
}

__global__ void ComputeLogisticActivations(float *fc2, int num_output_units, int batch_size, float *result)
{
    __shared__ float mySharedVariable[BLOCK_WIDTH];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_output_units * batch_size)
        return;
    mySharedVariable[threadIdx.x] = fc2[idx];
    // result[idx] = 1.0 / (1.0 + exp(-1.0 * fc2[idx]));
      result[idx] = 1.0 / (1.0 + exp(-1.0 * mySharedVariable[threadIdx.x]));
}

__global__ void LogisticLossBackprop(const float *label, int start_label, int my_class_size, int batch_size, float *diff)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= my_class_size * batch_size) return;
    int img_idx = idx / my_class_size;
    int tr_lab = static_cast<int>(label[img_idx]);
    if(tr_lab == start_label + idx % my_class_size){
        diff[idx] = diff[idx] - 1.0;
        // diff[idx] = -diff[idx] + 1.0;
    }
    // else{
    //     diff[idx] = -diff[idx];
    // }
}

__global__ void ReduceGradients(const float *fc1_conv, float *fc1_reduced, int size, int num_workers)
{
    __shared__ float mySharedVariable[BLOCK_WIDTH];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size) return;
    // fc1_reduced[idx] = 0;
    mySharedVariable[threadIdx.x] = 0;
    for(int i = 0; i < num_workers; i++){
        // int stride = i * len;
        // need to avoid this branching here
        // if(i == 0)
            // fc1_reduced[idx] = fc1_conv[idx + (i * len)];
        // else
            // fc1_reduced[idx] += fc1_conv[idx + (i * len)];
          mySharedVariable[threadIdx.x] += fc1_conv[idx + (i * size)];
    }
    fc1_reduced[idx] = mySharedVariable[threadIdx.x];
}

__global__ void ComputeLoss(const float *labels, const float *predictions, float *loss, int start_label, int my_class_size, int batch_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= my_class_size * batch_size) return;
    int img_idx = idx / my_class_size;
    int class_id = idx % my_class_size;
    // loss[img_idx] = 0;
    loss[img_idx + class_id] = -log(1 - predictions[img_idx + idx % my_class_size]);
    if(labels[img_idx] == start_label + idx % my_class_size)
        loss[img_idx + class_id] = -log(predictions[img_idx + idx % my_class_size]);
}

__global__ void ReduceLoss(const float *d_loss_all, float *d_loss_sum, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size) return;
    d_loss_sum[0] = 0;
    for(int i = 0; i < size; i++)
    { 
      // printf("from reduceloss: %f\n", d_loss_all[i]);
      d_loss_sum[0] += d_loss_all[i];
    }
    // d_loss_sum[0] /= size;
    // printf("d_loss_sum: %f\n", d_loss_sum[0]);
}

__global__ void ReduceLossToPlotVariable(const float *d_loss_all, float *d_loss_sum, int j, int size)
{
    d_loss_sum[j] = 0;
    for(int i = 0; i < size; i++)
    { 
      // printf("from reduceloss: %f\n", d_loss_all[i]);
      d_loss_sum[j] += d_loss_all[i];
    }
    // d_loss_sum[j] /= size;
}

// __global__ void ComputeAccuracy(const float *d_fc2smax, const float *d_labels_fc, float *test_set_accuracy, int size)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx >= size) return;
//     test_set_accuracy[0] = 0;
//     for(int i = 0; i < size; i++)
//     {
//         if(d_labels_fc[img ])
//     }
// }

__global__ void ArgMax(const float *predictions, float *returned_predictions, int my_class_size, int my_class, int j)
{
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int max_idx = -1;
    float max_value = -1;
    for(int i = 0; i < my_class_size; i++)
    {
        if(predictions[j * my_class_size + i] > max_value)
        {
            max_value = predictions[j * my_class_size + i];
            max_idx = i + my_class;
        }
    }
    returned_predictions[0] = (float)max_idx;
    returned_predictions[1] = max_value;
}

__global__ void ComputeZeroOneLoss(const float *predictions, const float *labels, float *accuracy, int num_workers)
{
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int max_idx = -1;
    float max_val = -1; 
    for(int i = 1; i <= num_workers; i *= 2)
    {
        if(predictions[i] > max_val)
        {
            max_val = predictions[i];
            max_idx = (int)predictions[i - 1];
        }
    }
    if(max_idx == labels[0])
        accuracy[0] += 1;
}

__global__ void PrintVal(float *a, int iter)
{
    printf("Iteration #: %d ====> loss: %f\n", iter, a[0]);
}

__global__ void PrintVec(float *a, int size)
{
    for(int i = 0; i < size; i++)
        printf("a[idx]: %f\n", a[i]);
}

__global__ void SetScalarVal(float *a, float val)
{
    a[0] = val;
}

int RoundUp(int nominator, int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

struct ConvBiasLayer
{
    int in_channels, out_channels, filter_size;
    int in_width, in_height, out_width, out_height;
    std::vector<float> pconv, pbias;

    ConvBiasLayer(int in_channels_, int out_channels_, int filter_size_,
                  int in_width_, int in_height_) : pconv(in_channels_ * filter_size_ * filter_size_ * out_channels_), pbias(out_channels_)
    {
        in_channels = in_channels_;
        out_channels = out_channels_;
        filter_size = filter_size_;
        in_width = in_width_;
        in_height = in_height_;
        out_width = in_width_ - filter_size_ + 1;
        out_height = in_height_ - filter_size_ + 1;
    }
};

struct MaxPoolLayer
{
    int size, stride;
    MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;
    FullyConnectedLayer(int inputs_, int outputs_) : inputs(inputs_), outputs(outputs_), pneurons(inputs_ * outputs_), pbias(outputs_) {}
};

struct TrainingContextConvBiasLayer
{
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor,
                             conv2Tensor, conv2BiasTensor, pool2Tensor;
    cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
    cudnnConvolutionDescriptor_t conv1Desc, conv2Desc;
    cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
    cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
    cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
    cudnnPoolingDescriptor_t poolDesc;

    int m_gpuid;
    int m_batchsize;
    size_t m_workspacesize;

    // TrainingContextConvBiasLayer& operator=(const TrainingContextConvBiasLayer&) = delete;
    // TrainingContextConvBiasLayer(const TrainingContextConvBiasLayer&) = delete;

    TrainingContextConvBiasLayer(int gpuid, int batch_size, ConvBiasLayer& conv1, MaxPoolLayer& pool1,
                                  ConvBiasLayer& conv2, MaxPoolLayer& pool2) : m_gpuid(gpuid), m_batchsize(batch_size)
    {
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCUDNN(cudnnCreate(&cudnnHandle));

        cudnnCreateTensorDescriptor((&dataTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));

        checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&conv2filterDesc));

        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));

        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));

        checkCUDNN(cudnnSetTensor4dDescriptor(conv1BiasTensor,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   1, conv1.out_channels,
                                   1, 1));
        checkCUDNN(cudnnSetTensor4dDescriptor(conv2BiasTensor,
                                  CUDNN_TENSOR_NCHW,
                                  CUDNN_DATA_FLOAT,
                                  1, conv2.out_channels,
                                  1, 1));

        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                   CUDNN_POOLING_MAX,
                                   CUDNN_PROPAGATE_NAN,
                                   pool1.size, pool1.size,
                                   0, 0,
                                   pool1.stride, pool1.stride));
        checkCUDNN(cudnnSetTensor4dDescriptor(pool2Tensor,
                                  CUDNN_TENSOR_NCHW,
                                  CUDNN_DATA_FLOAT,
                                  batch_size, conv2.out_channels,
                                  conv2.out_height / pool2.stride,
                                  conv2.out_width / pool2.stride));

         size_t workspace = 0;
         workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
         workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

         workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
         workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

         m_workspacesize = workspace;
    }

    ~TrainingContextConvBiasLayer()
    {
        checkCudaErrors(cudaSetDevice(m_gpuid));

        checkCudaErrors(cublasDestroy(cublasHandle));
        checkCUDNN(cudnnDestroy(cudnnHandle));
        checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv1BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(conv2BiasTensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(pool2Tensor));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    }

    size_t SetFwdConvolutionTensors(ConvBiasLayer& conv, cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                    cudnnConvolutionFwdAlgo_t& algo)
    {
        size_t sizeInBytes = 0;

        int n = m_batchsize;
        int c = conv.in_channels;
        int h = conv.in_height;
        int w = conv.in_width;

        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              n, c,
                                              h, w));

        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                              CUDNN_DATA_FLOAT,
                                              CUDNN_TENSOR_NCHW,
                                              conv.out_channels,
                                              conv.in_channels,
                                              conv.filter_size,
                                              conv.filter_size));
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                         0, 0,
                                         1, 1,
                                         1, 1,
                                         CUDNN_CROSS_CORRELATION,
                                         CUDNN_DATA_FLOAT));

         checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                          srcTensorDesc,
                                                          filterDesc,
                                                          &n, &c, &h, &w));

         checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               n, c,
                                               h, w));
         checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                        srcTensorDesc,
                                                        filterDesc,
                                                        convDesc,
                                                        dstTensorDesc,
                                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                        0,
                                                        &algo));

         checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                            srcTensorDesc,
                                                            filterDesc,
                                                            convDesc,
                                                            dstTensorDesc,
                                                            algo,
                                                            &sizeInBytes));
          return sizeInBytes;
    }

    size_t SetBwdConvolutionTensors(cudnnTensorDescriptor_t& srcTensorDesc, cudnnTensorDescriptor_t& dstTensorDesc,
                                    cudnnFilterDescriptor_t& filterDesc, cudnnConvolutionDescriptor_t& convDesc,
                                    cudnnConvolutionBwdFilterAlgo_t *falgo, cudnnConvolutionBwdDataAlgo_t *dalgo)
    {
          size_t sizeInBytes = 0, tmpsize = 0;
          if(falgo)
          {
              checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, falgo));

              checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                  cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
                  *falgo, &tmpsize));

              sizeInBytes = std::max(sizeInBytes, tmpsize);
          }
          if(dalgo)
          {
              checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, dalgo));

              checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                  cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
                  *dalgo, &tmpsize));

              sizeInBytes = std::max(sizeInBytes, tmpsize);
          }
          return sizeInBytes;
    }

    void ForwardPropagation(float *data, float *conv1, float *pool1, float *conv2, float *pool2,
                            float *pconv1, float *pconv1bias,
                            float *pconv2, float *pconv2bias, void *workspace)
    {
          float alpha = 1.0f, beta = 0.0f;
          checkCudaErrors(cudaSetDevice(m_gpuid));

          // Conv1 layer
          checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
                                             data, conv1filterDesc, pconv1, conv1Desc,
                                             conv1algo, workspace, m_workspacesize, &beta,
                                             conv1Tensor, conv1));
          checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv1BiasTensor,
                                    pconv1bias, &alpha, conv1Tensor, conv1));

          // Pool1 layer
          checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv1Tensor,
                                         conv1, &beta, pool1Tensor, pool1));

          // Conv2 layer
          checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, pool1Tensor,
                                             pool1, conv2filterDesc, pconv2, conv2Desc,
                                             conv2algo, workspace, m_workspacesize, &beta,
                                             conv2Tensor, conv2));
          checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, conv2BiasTensor,
                                    pconv2bias, &alpha, conv2Tensor, conv2));

          // Pool2 layer
          checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, conv2Tensor,
                                         conv2, &beta, pool2Tensor, pool2));
    }

    void Backpropagation(ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
                         float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2,
                         float *pconv1, float *pconv1bias,
                         float *pconv2, float *pconv2bias,
                         float *gconv1, float *gconv1bias, float *dpool1,
                         float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
                         float *dfc1,
                         void *workspace)
    {
          float alpha = 1.0f, beta = 0.0f;
          // Pool2 layer
          checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                                          pool2Tensor, pool2, pool2Tensor, dfc1,
                                          conv2Tensor, conv2, &beta, conv2Tensor, dpool2));

          // Conv2 layer
          checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv2Tensor,
                                                  dpool2, &beta, conv2BiasTensor, gconv2bias));


          checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, pool1Tensor,
                                                    pool1, conv2Tensor, dpool2, conv2Desc,
                                                    conv2bwfalgo, workspace, m_workspacesize,
                                                    &beta, conv2filterDesc, gconv2));

          checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, conv2filterDesc,
                                                  pconv2, conv2Tensor, dpool2, conv2Desc,
                                                  conv2bwdalgo, workspace, m_workspacesize,
                                                  &beta, pool1Tensor, dconv2));

          // Pool1 layer
          checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha,
                                          pool1Tensor, pool1, pool1Tensor, dconv2,
                                          conv1Tensor, conv1, &beta, conv1Tensor, dpool1));

          // Conv1 layer
          checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, conv1Tensor,
                                                  dpool1, &beta, conv1BiasTensor, gconv1bias));

          checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, dataTensor,
                                                    data, conv1Tensor, dpool1, conv1Desc,
                                                    conv1bwfalgo, workspace, m_workspacesize,
                                                    &beta, conv1filterDesc, gconv1));

          // No need for convBackwardData because there are no more layers below
    }

    void UpdateWeights(float step_size, int num_workers, ConvBiasLayer& conv1, ConvBiasLayer& conv2, float *pconv1, float *pconv1bias,
                      float *pconv2, float *pconv2bias, float *gconv1, float *gconv1bias, float *gconv2, float *gconv2bias)
    {
          float alpha = -step_size / num_workers;
          checkCudaErrors(cudaSetDevice(m_gpuid));

          // Conv1
          checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pconv.size()),
                                    &alpha, gconv1, 1, pconv1, 1));
          checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv1.pbias.size()),
                                      &alpha, gconv1bias, 1, pconv1bias, 1));

          // Conv2
          checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pconv.size()),
                                    &alpha, gconv2, 1, pconv2, 1));
          checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(conv2.pbias.size()),
                                      &alpha, gconv2bias, 1, pconv2bias, 1));
    }
};

struct TrainingContextFullyConnectedLayer
{
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t fc1Tensor, fc2Tensor;
    cudnnActivationDescriptor_t fc1Activation;

    int m_gpuid;
    int m_batchsize;

    FullyConnectedLayer& ref_fc1, &ref_fc2;
    // Disable copying
    TrainingContextFullyConnectedLayer& operator=(const TrainingContextFullyConnectedLayer&) = delete;
    TrainingContextFullyConnectedLayer(const TrainingContextFullyConnectedLayer&) = delete;

    TrainingContextFullyConnectedLayer(int gpuid, int batch_size, FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid), m_batchsize(batch_size)
    {
        // Create CUBLAS and CUDNN handles
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCUDNN(cudnnCreate(&cudnnHandle));

        // Create tensor descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc1Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, fc1.outputs, 1, 1));

        checkCUDNN(cudnnSetTensor4dDescriptor(fc2Tensor,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size, fc2.outputs, 1, 1));

        checkCUDNN(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN, 0.0));
    }

    ~TrainingContextFullyConnectedLayer()
    {
        checkCudaErrors(cudaSetDevice(m_gpuid));

        checkCudaErrors(cublasDestroy(cublasHandle));
        checkCUDNN(cudnnDestroy(cudnnHandle));

        checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
        checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
    }

    void ForwardPropagationFullyConnectedLayer(float *pool2, float *fc1, float *fc1relu,
                                               float *pfc1, float *pfc1bias, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(m_gpuid));

        // FC1 layer
        // Forward propagate neurons using weights (fc1 = pfc1'*pool2)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_fc1.outputs, m_batchsize, ref_fc1.inputs,
                                    &alpha,
                                    pfc1, ref_fc1.inputs,
                                    pool2, ref_fc1.inputs,
                                    &beta,
                                    fc1, ref_fc1.outputs));
        // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_fc1.outputs, m_batchsize, 1,
                                    &alpha,
                                    pfc1bias, ref_fc1.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc1, ref_fc1.outputs));

        // ReLU activation
        checkCUDNN(cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
                                          fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));
    }

    void ForwardPropagationLogisticLayer(float *fc1relu, float *fc2, float *result, float *pfc2, float *pfc2bias, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;
        // FC2 layer
        // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_fc2.outputs, m_batchsize, ref_fc2.inputs,
                                    &alpha,
                                    pfc2, ref_fc2.inputs,
                                    fc1relu, ref_fc2.inputs,
                                    &beta,
                                    fc2, ref_fc2.outputs));
        // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_fc2.outputs, m_batchsize, 1,
                                    &alpha,
                                    pfc2bias, ref_fc2.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc2, ref_fc2.outputs));
        ComputeLogisticActivations<<<RoundUp(m_batchsize * ref_fc2.outputs, BLOCK_WIDTH), BLOCK_WIDTH>>>(fc2, ref_fc2.outputs, m_batchsize, result);

    }

    void ForwardPropagationSoftmaxLayer(float *fc1relu, float *fc2, float *result, float *pfc2, float *pfc2bias, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(m_gpuid));
        // FC2 layer
        // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ref_fc2.outputs, m_batchsize, ref_fc2.inputs,
                                    &alpha,
                                    pfc2, ref_fc2.inputs,
                                    fc1relu, ref_fc2.inputs,
                                    &beta,
                                    fc2, ref_fc2.outputs));
        // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    ref_fc2.outputs, m_batchsize, 1,
                                    &alpha,
                                    pfc2bias, ref_fc2.outputs,
                                    onevec, 1,
                                    &alpha,
                                    fc2, ref_fc2.outputs));

        // Softmax loss
        checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
    }

    void ForwardPropagationSoftmax(float *fc2, float *result)
    {
        float alpha = 1.0f, beta = 0.0f;
        // Softmax loss
        checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
    }

    void BackpropagationLogisticLayer(int start_label, float *labels, float *fc1relu, float *fc2, float *fc2smax, float *dloss_data,
                                      float *pfc2, float *pfc2bias, float *gfc2, float *gfc2bias,
                                      float *dfc2, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;
        float scalVal = 1.0f / static_cast<float>(m_batchsize);
        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Initialization (using the training error function)
        checkCudaErrors(cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchsize * ref_fc2.outputs, cudaMemcpyDeviceToDevice));
        // Softmax layer
        LogisticLossBackprop<<<RoundUp(m_batchsize * ref_fc2.outputs, BLOCK_WIDTH), BLOCK_WIDTH>>>(labels, start_label, ref_fc2.outputs, m_batchsize, dloss_data);
        // Accounting for batch size in SGD
        checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchsize, &scalVal, dloss_data, 1));

        // FC2 layer
        // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchsize,
                                    &alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs)); // Verify
        // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, m_batchsize,
                                    &alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10 * 10xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, m_batchsize, ref_fc2.outputs,
                                    &alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));
    }

    void BackpropagationFullyConnectedLayer(float *pool2, float *fc1, float *fc1relu, float *dfc1, float*dfc1relu, float *dfc2,
                                            float *pfc1, float *pfc1bias, float *gfc1, float *gfc1bias, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;
        // ReLU activation
        checkCUDNN(cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
                                           fc1Tensor, fc1relu, fc1Tensor, dfc2,
                                           fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));
        // FC1 layer
        // Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchsize,
                                    &alpha, pool2, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
        // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
        checkCudaErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, m_batchsize,
                                    &alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, m_batchsize, ref_fc1.outputs,
                                    &alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));
    }

    void UpdateWeights(float step_size, int num_workers,
                       float *pfc1, float *pfc1bias,
                       float *pfc2, float *pfc2bias,
                       float *gfc1, float *gfc1bias,
                       float *gfc2, float *gfc2bias)
   {
       float alpha = -step_size * num_workers;
       checkCudaErrors(cudaSetDevice(m_gpuid));
       // Fully connected 1
       checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
                                   &alpha, gfc1, 1, pfc1, 1));
       checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
                                   &alpha, gfc1bias, 1, pfc1bias, 1));

       // Fully connected 2
       checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
                                   &alpha, gfc2, 1, pfc2, 1));
       checkCudaErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
                                   &alpha, gfc2bias, 1, pfc2bias, 1));
   }
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int num_processors, proc_id;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    // printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, proc_id, num_processors);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    // printf("rank %d ==> num_gpus %d\n", proc_id, num_gpus);

    // int gpu_rank = proc_id % num_gpus;
    int gpu_rank = proc_id % NUM_WORKERS_DATA_PARALLELISM;
    // checkCudaErrors(cudaSetDevice(gpu_rank));
    printf("proc_id %d linked to device %d\n", proc_id, gpu_rank);

    if(proc_id < NUM_WORKERS_DATA_PARALLELISM)
    {   
        int count_req, len;

        checkCudaErrors(cudaSetDevice(gpu_rank));
        size_t img_width, img_height, img_channels;
        // size_t train_size, test_size;
        printf("Reading input images\n");
        img_channels = 1;
        size_t train_size = ReadUByteDataset(TRAIN_IMAGES, TRAIN_LABELS, nullptr, nullptr, img_width, img_height);
        size_t test_size = ReadUByteDataset(TEST_IMAGES, TEST_LABELS, nullptr, nullptr, img_width, img_height);

        printf("training_size: %d\n", train_size);
        printf("training_size: %d\n", test_size);

        std::vector<uint8_t> train_images(train_size * img_width * img_height * img_channels);
        std::vector<uint8_t> train_labels(train_size);
        std::vector<uint8_t> test_images(test_size * img_width * img_height * img_channels);
        std::vector<uint8_t> test_labels(test_size);

        if (ReadUByteDataset(TRAIN_IMAGES, TRAIN_LABELS, &train_images[0], &train_labels[0], img_width, img_height) != train_size)
            return 2;
        if (ReadUByteDataset(TEST_IMAGES, TEST_LABELS, &test_images[0], &test_labels[0], img_width, img_height) != test_size)
            return 3;

        ConvBiasLayer conv1((int)img_channels, CONV1_FILTERS, CONV1_FILTER_SIZE, (int)img_width, (int)img_height);
        MaxPoolLayer pool1(2, 2);
        ConvBiasLayer conv2(CONV1_FILTERS, CONV2_FILTERS, CONV2_FILTER_SIZE, conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
        MaxPoolLayer pool2(2, 2);
        printf("conv1.out_width: %d; conv1.out_height: %d\n", conv1.out_width, conv1.out_height);
        printf("conv1.out_width / pool1.stride: %d; conv1.out_height / pool1.stride: %d\n", conv1.out_width / pool1.stride, conv1.out_height / pool1.stride);
        printf("img_height: %d; img_width: %d\n", img_height, img_height);
        TrainingContextConvBiasLayer conv_context(gpu_rank, BATCH_SIZE, conv1, pool1, conv2, pool2);

        // Create random network
        std::random_device rd;
        std::mt19937 gen(RANDOM_SEED < 0 ? rd() : static_cast<unsigned int>(RANDOM_SEED));

        // Xavier weight filling
        float wconv1 = sqrt(3.0f / (conv1.filter_size * conv1.filter_size * conv1.in_channels));
        std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
        float wconv2 = sqrt(3.0f / (conv2.filter_size * conv2.filter_size * conv2.in_channels));
        std::uniform_real_distribution<> dconv2(-wconv2, wconv2);

        // Randomize network
        for (auto&& iter : conv1.pconv)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv1.pbias)
            iter = static_cast<float>(dconv1(gen));
        for (auto&& iter : conv2.pconv)
            iter = static_cast<float>(dconv2(gen));
        for (auto&& iter : conv2.pbias)
            iter = static_cast<float>(dconv2(gen));

        // Forward propagation data
        float *d_data, *d_labels, *d_conv1, *d_pool1, *d_conv2, *d_pool2;
        //                         Buffer    | Element       | N                   | C                  | H                                 | W
        //-----------------------------------------------------------------------------------------------------------------------------------------
        checkCudaErrors(cudaMalloc(&d_data,    sizeof(float) * conv_context.m_batchsize * img_channels           * img_height                            * img_width));
        checkCudaErrors(cudaMalloc(&d_labels,  sizeof(float) * conv_context.m_batchsize * 1                  * 1                                 * 1));
        checkCudaErrors(cudaMalloc(&d_conv1,   sizeof(float) * conv_context.m_batchsize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
        checkCudaErrors(cudaMalloc(&d_pool1,   sizeof(float) * conv_context.m_batchsize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
        checkCudaErrors(cudaMalloc(&d_conv2,   sizeof(float) * conv_context.m_batchsize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
        checkCudaErrors(cudaMalloc(&d_pool2,   sizeof(float) * conv_context.m_batchsize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));

        float *d_test_data, *d_test_labels, *d_conv1_test, *d_pool1_test, *d_conv2_test, *d_pool2_test;
        checkCudaErrors(cudaMalloc(&d_test_data, sizeof(float) * test_size * img_channels * img_height * img_width));
        checkCudaErrors(cudaMalloc(&d_test_labels, sizeof(float) * test_size));
        checkCudaErrors(cudaMalloc(&d_conv1_test,   sizeof(float) * test_size * conv1.out_channels * conv1.out_height                  * conv1.out_width));
        checkCudaErrors(cudaMalloc(&d_pool1_test,   sizeof(float) * test_size * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
        checkCudaErrors(cudaMalloc(&d_conv2_test,   sizeof(float) * test_size * conv2.out_channels * conv2.out_height                  * conv2.out_width));
        checkCudaErrors(cudaMalloc(&d_pool2_test,   sizeof(float) * test_size * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));

        // Network parameters
        float *d_pconv1, *d_pconv1bias, *d_pconv2, *d_pconv2bias;
        checkCudaErrors(cudaMalloc(&d_pconv1,     sizeof(float) * conv1.pconv.size()));
        checkCudaErrors(cudaMalloc(&d_pconv1bias, sizeof(float) * conv1.pbias.size()));
        checkCudaErrors(cudaMalloc(&d_pconv2,     sizeof(float) * conv2.pconv.size()));
        checkCudaErrors(cudaMalloc(&d_pconv2bias, sizeof(float) * conv2.pbias.size()));

        // Network parameter gradients
        float *d_gconv1, *d_gconv1bias, *d_gconv2, *d_gconv2bias;
        checkCudaErrors(cudaMalloc(&d_gconv1,     sizeof(float) * conv1.pconv.size()));
        checkCudaErrors(cudaMalloc(&d_gconv1bias, sizeof(float) * conv1.pbias.size()));
        checkCudaErrors(cudaMalloc(&d_gconv2,     sizeof(float) * conv2.pconv.size()));
        checkCudaErrors(cudaMalloc(&d_gconv2bias, sizeof(float) * conv2.pbias.size()));

        float *d_dpool1, *d_dpool2, *d_dconv2;
        checkCudaErrors(cudaMalloc(&d_dpool1,   sizeof(float) * conv_context.m_batchsize * conv1.out_channels * conv1.out_height                  * conv1.out_width));
        checkCudaErrors(cudaMalloc(&d_dpool2,   sizeof(float) * conv_context.m_batchsize * conv2.out_channels * conv2.out_height                  * conv2.out_width));
        checkCudaErrors(cudaMalloc(&d_dconv2,   sizeof(float) * conv_context.m_batchsize * conv1.out_channels * (conv1.out_height / pool1.stride) * (conv1.out_width / pool1.stride)));
        // checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * conv_context.m_batchsize * fc.inputs));

        float *d_dfc1_conv, *d_dfc1_reduced;
        checkCudaErrors(cudaMalloc(&d_dfc1_conv, sizeof(float) * conv_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));
        checkCudaErrors(cudaMalloc(&d_dfc1_reduced, sizeof(float) * conv_context.m_batchsize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride)));

        float *d_gconv1_all, *d_gconv1bias_all, *d_gconv2_all, *d_gconv2bias_all;
        checkCudaErrors(cudaMalloc(&d_gconv1_all,     sizeof(float) * NUM_WORKERS_DATA_PARALLELISM * conv1.pconv.size()));
        checkCudaErrors(cudaMalloc(&d_gconv1bias_all, sizeof(float) * NUM_WORKERS_DATA_PARALLELISM * conv1.pbias.size()));
        checkCudaErrors(cudaMalloc(&d_gconv2_all,     sizeof(float) * NUM_WORKERS_DATA_PARALLELISM * conv2.pconv.size()));
        checkCudaErrors(cudaMalloc(&d_gconv2bias_all, sizeof(float) * NUM_WORKERS_DATA_PARALLELISM * conv2.pbias.size()));

        // Temporary buffers and workspaces
        // float *d_onevec;
        void *d_cudnn_workspace = nullptr;
        // cudaMalloc(&d_onevec, sizeof(float)* conv_context.m_batchsize);
        if (conv_context.m_workspacesize > 0)
            checkCudaErrors(cudaMalloc(&d_cudnn_workspace, conv_context.m_workspacesize));

        // Copy initial network to device
        checkCudaErrors(cudaMemcpyAsync(d_pconv1, &conv1.pconv[0],     sizeof(float) * conv1.pconv.size(),  cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pconv1bias, &conv1.pbias[0], sizeof(float) * conv1.pbias.size(),  cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pconv2, &conv2.pconv[0],     sizeof(float) * conv2.pconv.size(),  cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pconv2bias, &conv2.pbias[0], sizeof(float) * conv2.pbias.size(),  cudaMemcpyHostToDevice));

        // Fill one-vector with ones
        // FillOnes<<<RoundUp(conv_context.m_batchsize, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_onevec, conv_context.m_batchsize);

        printf("Preparing dataset\n");

        // Normalize training set to be in [0,1]
        std::vector<float> train_images_float(train_images.size()), train_labels_float(train_size);
        for (size_t i = 0; i < train_size * img_channels * img_width * img_height; ++i)
            train_images_float[i] = (float)train_images[i] / 255.0f;

        for (size_t i = 0; i < train_size; ++i)
            train_labels_float[i] = (float)train_labels[i];

        std::vector<float> test_images_float(test_images.size()), test_labels_float(test_size);
        for (size_t i = 0; i < test_size * img_channels * img_width * img_height; ++i)
            test_images_float[i] = (float)test_images[i] / 255.0f;

        for (size_t i = 0; i < test_size; ++i)
            test_labels_float[i] = (float)test_labels[i];

        checkCudaErrors(cudaMemcpyAsync(d_test_data, &test_images_float[0], sizeof(float) * test_size * img_channels * img_height * img_width, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_test_labels, &test_labels_float[0], sizeof(float) * test_size, cudaMemcpyHostToDevice));

        printf("Training...\n");

        // Use SGD to train the network
        checkCudaErrors(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        for(int iter = 0; iter < ITERATIONS; ++iter)
        {
            checkCudaErrors(cudaSetDevice(gpu_rank));
            // Train
            printf("iteration #%d\n", iter);

            // if((iter + 1) % TEST_INTERVAL == 0)
            // {
            //   if(proc_id == 0)
            //   {
            //       conv_context.ForwardPropagation(d_test_data, d_conv1_test, d_pool1_test, d_conv2_test, d_pool2_test, d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_cudnn_workspace);
            //       MPI_Request req[NUM_WORKERS_DATA_PARALLELISM];
            //       MPI_Request req_labels[NUM_WORKERS_DATA_PARALLELISM];
            //       // MPI_Status stat[NUM_WORKERS_DATA_PARALLELISM];

            //       for(int i=0;i < NUM_WORKERS_DATA_PARALLELISM; i++){
            //         int curr_FC_addr = i + NUM_WORKERS_DATA_PARALLELISM;
            //         MPI_Isend(d_pool2_test, test_size * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride), MPI_FLOAT, curr_FC_addr, FORPROP_TAG, MPI_COMM_WORLD, &req[i]);
            //         MPI_Isend(d_test_labels, test_size, MPI_FLOAT, curr_FC_addr, FORPROP_LABELS_TAG, MPI_COMM_WORLD, &req_labels[i]);
            //       }

            //       MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req, MPI_STATUSES_IGNORE);
            //       MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req_labels, MPI_STATUSES_IGNORE);

            //       int flag = 1;
            //       MPI_Request flag_sends[NUM_WORKERS_DATA_PARALLELISM - 1];
            //       for(int i = 1; i < NUM_WORKERS_DATA_PARALLELISM; i++)
            //           MPI_Isend(&flag, 1, MPI_INT, i, FLAG_TAG, MPI_COMM_WORLD, &flag_sends[i - 1]);
            //       // printf("proc_id: %d ---> sent!\n", proc_id);

            //       MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, flag_sends, MPI_STATUSES_IGNORE);
            //       // printf("proc_id: %d ---> waited!\n", proc_id);
            //   }
            //   else
            //   {   
            //       int rec_flag;
            //       MPI_Recv(&rec_flag, 1, MPI_INT, 0, FLAG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //   }
            // }

            int imageid = iter % (train_size / conv_context.m_batchsize);
            // Prepare current batch on device
            // printf("Allocating dataset variables in proc_id: %d and gpu_id: %d\n", proc_id, gpu_rank);
            checkCudaErrors(cudaMemcpyAsync(d_data, &train_images_float[imageid * conv_context.m_batchsize * img_width*img_height*img_channels],
                                        sizeof(float) * conv_context.m_batchsize * img_channels * img_width * img_height, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_labels, &train_labels_float[imageid * conv_context.m_batchsize],
                                            sizeof(float) * conv_context.m_batchsize, cudaMemcpyHostToDevice));
            // printf("Allocated dataset variables in proc_id: %d and gpu_id: %d\n", proc_id, gpu_rank);

            // Forward propagation
            // Maybe put MPI_Waitall here?
            conv_context.ForwardPropagation(d_data, d_conv1, d_pool1, d_conv2, d_pool2, d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_cudnn_workspace);
            // Check the type of send; Also, need to change the following to handle generalized neural nets
            // In scheme 'a', we are going to send a batch of activations with the leadind dimension as the batch_size to the fullyconnected layer workers
            MPI_Request req[NUM_WORKERS_DATA_PARALLELISM];
            MPI_Request req_labels[NUM_WORKERS_DATA_PARALLELISM];
            // MPI_Status stat[NUM_WORKERS_DATA_PARALLELISM];

            for(int i=0;i<NUM_WORKERS_DATA_PARALLELISM;i++){
              int curr_FC_addr = i + NUM_WORKERS_DATA_PARALLELISM;
              MPI_Isend(d_pool2, conv_context.m_batchsize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride), MPI_FLOAT, curr_FC_addr, FORPROP_TAG, MPI_COMM_WORLD, &req[i]);
              MPI_Isend(d_labels, conv_context.m_batchsize, MPI_FLOAT, curr_FC_addr, FORPROP_LABELS_TAG, MPI_COMM_WORLD, &req_labels[i]);
            }

            // MPI_Isend(d_pool2, conv_context.m_batchsize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride), MPI_FLOAT, FC_ADDRESS_2, FORPROP_TAG, MPI_COMM_WORLD, &req[1]);

            // shouldnt this be MPI_STATUS_IGNORE? STATUSES is also fine
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req, MPI_STATUSES_IGNORE);
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req_labels, MPI_STATUSES_IGNORE);

            MPI_Request req_recv[NUM_WORKERS_DATA_PARALLELISM];
            for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv_context.m_batchsize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride);
                // MPI_Recv(d_dfc1_conv + (i * len), len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, BACKPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // printf("conv size: %d; batch_size: %d\n", conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride), conv_context.m_batchsize);
                MPI_Irecv(d_dfc1_conv + (i * len), len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, BACKPROP_TAG, MPI_COMM_WORLD, &req_recv[i]);
                // printf("fc proc_id: %d ====> ISend started to address: %d\n", proc_id, i + NUM_WORKERS_DATA_PARALLELISM);
            }
            // printf("conv proc_id: %d ====> send waiting\n", proc_id);
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req_recv, MPI_STATUSES_IGNORE);
            // printf("conv proc_id: %d ====> send waited\n", proc_id);

            len = conv_context.m_batchsize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride);
            ReduceGradients<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_dfc1_conv, d_dfc1_reduced, len, NUM_WORKERS_DATA_PARALLELISM);
            // ConvBiasLayer& layer_conv1, MaxPoolLayer& layer_pool1, ConvBiasLayer& layer_conv2, MaxPoolLayer& layer_pool2,
            //  float *data, float *labels, float *conv1, float *pool1, float *conv2, float *pool2,
            //  float *pconv1, float *pconv1bias,
            //  float *pconv2, float *pconv2bias,
            //  float *gconv1, float *gconv1bias, float *dpool1,
            //  float *gconv2, float *gconv2bias, float *dconv2, float *dpool2,
            //  float *dfc1,
            //  void *workspace
            // printf("Conv proc_id: %d ====> backprop started\n", proc_id);
            conv_context.Backpropagation(conv1, pool1, conv2, pool2, d_data, d_labels, d_conv1, d_pool1, d_conv2, d_pool2,
                                         d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias, d_gconv1, d_gconv1bias, d_dpool1,
                                         d_gconv2, d_gconv2bias, d_dconv2, d_dpool2, d_dfc1_reduced, d_cudnn_workspace);
            // printf("Conv proc_id: %d ====> backprop done\n", proc_id);

           MPI_Request req_conv1[NUM_WORKERS_DATA_PARALLELISM - 1];
           count_req = 0;
           for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv1.pconv.size();
                if(i == proc_id) continue;
                MPI_Isend(d_gconv1, len, MPI_FLOAT, i, SYNC_CONV1_TAG, MPI_COMM_WORLD, &req_conv1[count_req]);
                count_req++;
           }
           MPI_Request req_conv1_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
           count_req = 0;
           for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv1.pconv.size();
                if(i == proc_id)
                    copyData<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_gconv1, d_gconv1_all + (i * len), len);
                else
                {
                    // MPI_Recv(d_gconv1_all + (i * len), len, MPI_FLOAT, i, SYNC_CONV1_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Irecv(d_gconv1_all + (i * len), len, MPI_FLOAT, i, SYNC_CONV1_TAG, MPI_COMM_WORLD, &req_conv1_recv[count_req]);
                    count_req++;
                }
           }
           MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_conv1, MPI_STATUSES_IGNORE);
           MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_conv1_recv, MPI_STATUSES_IGNORE);
           // const float *fc1_conv, float *fc1_reduced, int size, int num_workers
           len = conv1.pconv.size();
           ReduceGradients<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_gconv1_all, d_gconv1, len, NUM_WORKERS_DATA_PARALLELISM);

           MPI_Request req_conv1bias[NUM_WORKERS_DATA_PARALLELISM - 1];
           count_req = 0;
           for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv1.pbias.size();
                if(i == proc_id) continue;
                MPI_Isend(d_gconv1bias, len, MPI_FLOAT, i, SYNC_CONV1BIAS_TAG, MPI_COMM_WORLD, &req_conv1bias[count_req]);
                count_req++;
           }
           MPI_Request req_conv1bias_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
           count_req = 0;
           for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv1.pbias.size();
                if(i == proc_id)
                    copyData<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_gconv1bias, d_gconv1bias_all + (i * len), len);
                else
                {
                    // MPI_Recv(d_gconv1bias_all + (i * len), len, MPI_FLOAT, i, SYNC_CONV1BIAS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Irecv(d_gconv1bias_all + (i * len), len, MPI_FLOAT, i, SYNC_CONV1BIAS_TAG, MPI_COMM_WORLD, &req_conv1bias_recv[count_req]);
                    count_req++;
                }
           }
           MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_conv1bias, MPI_STATUSES_IGNORE);
           MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_conv1bias_recv, MPI_STATUSES_IGNORE);
           // const float *fc1_conv, float *fc1_reduced, int size, int num_workers
           len = conv1.pbias.size();
           ReduceGradients<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_gconv1bias_all, d_gconv1bias, len, NUM_WORKERS_DATA_PARALLELISM);

           MPI_Request req_conv2[NUM_WORKERS_DATA_PARALLELISM - 1];
           count_req = 0;
           for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv2.pconv.size();
                if(i == proc_id) continue;
                MPI_Isend(d_gconv2, len, MPI_FLOAT, i, SYNC_CONV2_TAG, MPI_COMM_WORLD, &req_conv2[count_req]);
                count_req++;
           }
           MPI_Request req_conv2_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
           count_req = 0;
           for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv2.pconv.size();
                if(i == proc_id)
                    copyData<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_gconv2, d_gconv2_all + (i * len), len);
                else
                {
                    // MPI_Recv(d_gconv2_all + (i * len), len, MPI_FLOAT, i, SYNC_CONV2_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Irecv(d_gconv2_all + (i * len), len, MPI_FLOAT, i, SYNC_CONV2_TAG, MPI_COMM_WORLD, &req_conv2_recv[count_req]);
                    count_req++;
                }
           }
           MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_conv2, MPI_STATUSES_IGNORE);
           MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_conv2_recv, MPI_STATUSES_IGNORE);
           // const float *fc1_conv, float *fc1_reduced, int size, int num_workers
           len = conv2.pconv.size();
           ReduceGradients<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_gconv2_all, d_gconv2, len, NUM_WORKERS_DATA_PARALLELISM);

           MPI_Request req_conv2bias[NUM_WORKERS_DATA_PARALLELISM - 1];
           count_req = 0;
           for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv2.pbias.size();
                if(i == proc_id) continue;
                MPI_Isend(d_gconv2bias, len, MPI_FLOAT, i, SYNC_CONV2BIAS_TAG, MPI_COMM_WORLD, &req_conv2bias[count_req]);
                count_req++;
           }
           MPI_Request req_conv2bias_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
           count_req = 0;
           for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = conv2.pbias.size();
                if(i == proc_id)
                    copyData<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_gconv2bias, d_gconv2bias_all + (i * len), len);
                else
                {
                    // MPI_Recv(d_gconv2bias_all + (i * len), len, MPI_FLOAT, i, SYNC_CONV2BIAS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Irecv(d_gconv2bias_all + (i * len), len, MPI_FLOAT, i, SYNC_CONV2BIAS_TAG, MPI_COMM_WORLD, &req_conv2bias_recv[count_req]);
                    count_req++;
                }
           }
           MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_conv2bias, MPI_STATUSES_IGNORE);
           MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_conv2bias_recv, MPI_STATUSES_IGNORE);
           // const float *fc1_conv, float *fc1_reduced, int size, int num_workers
           len = conv2.pbias.size();
           ReduceGradients<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_gconv2bias_all, d_gconv2bias, len, NUM_WORKERS_DATA_PARALLELISM);

           // float step_size, int num_workers, ConvBiasLayer& conv1, ConvBiasLayer& conv2, float *pconv1, float *pconv1bias,
           // float *pconv2, float *pconv2bias, float *gconv1, float *gconv1bias, float *gconv2, float *gconv2bias
           conv_context.UpdateWeights(STEP_SIZE, NUM_WORKERS_DATA_PARALLELISM, conv1, conv2, d_pconv1, d_pconv1bias, d_pconv2, d_pconv2bias,
                                      d_gconv1, d_gconv1bias, d_gconv2, d_gconv2bias);
           // printf("Conv proc_id: %d ====> updated weights\n", proc_id);
        }
    }
    if(proc_id >= NUM_WORKERS_DATA_PARALLELISM && proc_id < num_processors)
    {
        int len, count_req;
        // printf("proc_id: %d\n", proc_id);
        checkCudaErrors(cudaSetDevice(gpu_rank));
        int my_class_size, my_class;
        // Check the dimensions
        // FullyConnectedLayer fc1(CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM * CONV2_FILTERS, FC1_OUTPUT_DIM_FRACTION);
        // FullyConnectedLayer fc2(FC1_OUTPUT_DIM, FC2_OUTPUT_DIM_FRACTION);
        // Num of activations should be divisible by 12 (num_workers = 2,3,4)
        FullyConnectedLayer fc1(CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM * CONV2_FILTERS, FC1_OUTPUT_DIM/NUM_WORKERS_DATA_PARALLELISM); // Divisible by 12
        printf("proc_id: %d has created fc1 reference\n", proc_id);
        if(proc_id != 2*NUM_WORKERS_DATA_PARALLELISM - 1){
            my_class_size = ceil(10.0/NUM_WORKERS_DATA_PARALLELISM);
            my_class = ceil(10.0/NUM_WORKERS_DATA_PARALLELISM) * (proc_id - NUM_WORKERS_DATA_PARALLELISM);
        }
        else{
            my_class_size = 10 - ceil(10.0/NUM_WORKERS_DATA_PARALLELISM) * (NUM_WORKERS_DATA_PARALLELISM - 1);
            my_class = ceil(10.0/NUM_WORKERS_DATA_PARALLELISM) * (proc_id - NUM_WORKERS_DATA_PARALLELISM);
        }
        int f_class_size = ceil(10.0/NUM_WORKERS_DATA_PARALLELISM);
        int end_class_size = 10 - ceil(10.0/NUM_WORKERS_DATA_PARALLELISM) * (NUM_WORKERS_DATA_PARALLELISM - 1);
        printf("proc_id: %d ===> my_class_size is %d and my_class is %d\n", proc_id, (int)my_class_size, (int)my_class);

        FullyConnectedLayer fc2(FC1_OUTPUT_DIM, (int)my_class_size);
        printf("proc_id: %d has created fc2 reference\n", proc_id);

        TrainingContextFullyConnectedLayer fc_context(gpu_rank, BATCH_SIZE * NUM_WORKERS_DATA_PARALLELISM, fc1, fc2);

        // Create random network
        std::random_device rd;
        std::mt19937 gen(RANDOM_SEED < 0 ? rd() : static_cast<unsigned int>(RANDOM_SEED));

        float wfc1 = sqrt(3.0f / (fc1.inputs * FC1_OUTPUT_DIM));
        std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
        float wfc2 = sqrt(3.0f / (fc2.inputs * FC2_OUTPUT_DIM));
        std::uniform_real_distribution<> dfc2(-wfc2, wfc2);

        for (auto&& iter : fc1.pneurons)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc1.pbias)
            iter = static_cast<float>(dfc1(gen));
        for (auto&& iter : fc2.pneurons)
            iter = static_cast<float>(dfc2(gen));
        for (auto&& iter : fc2.pbias)
            iter = static_cast<float>(dfc2(gen));

        float *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax, *f_class;
        checkCudaErrors(cudaMalloc(&d_fc1,     sizeof(float) * fc_context.m_batchsize * fc1.outputs));
        checkCudaErrors(cudaMalloc(&d_fc1relu, sizeof(float) * fc_context.m_batchsize * fc1.outputs));
        checkCudaErrors(cudaMalloc(&d_fc2,     sizeof(float) * fc_context.m_batchsize * fc2.outputs));
        checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float) * fc_context.m_batchsize * fc2.outputs));
        checkCudaErrors(cudaMalloc(&f_class, sizeof(int)));
        checkCudaErrors(cudaMemcpyAsync(f_class, &my_class, sizeof(int), cudaMemcpyHostToDevice));
        // checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM));

        float *d_pfc1, *d_pfc1bias, *d_pfc2, *d_pfc2bias;
        checkCudaErrors(cudaMalloc(&d_pfc1,       sizeof(float) * fc1.pneurons.size()));
        checkCudaErrors(cudaMalloc(&d_pfc1bias,   sizeof(float) * fc1.pbias.size()));
        checkCudaErrors(cudaMalloc(&d_pfc2,       sizeof(float) * fc2.pneurons.size()));
        checkCudaErrors(cudaMalloc(&d_pfc2bias,   sizeof(float) * fc2.pbias.size()));

        float *d_gfc1, *d_gfc1bias, *d_gfc2, *d_gfc2bias;
        checkCudaErrors(cudaMalloc(&d_gfc1,       sizeof(float) * fc1.pneurons.size()));
        checkCudaErrors(cudaMalloc(&d_gfc1bias,   sizeof(float) * fc1.pbias.size()));
        checkCudaErrors(cudaMalloc(&d_gfc2,       sizeof(float) * fc2.pneurons.size()));
        checkCudaErrors(cudaMalloc(&d_gfc2bias,   sizeof(float) * fc2.pbias.size()));

        float *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata, *d_dfc2_t;
        checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * fc_context.m_batchsize * fc1.inputs));
        checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float) * fc_context.m_batchsize * fc1.outputs));
        // checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM));
        // checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM));

        checkCudaErrors(cudaMalloc(&d_dfc2,     sizeof(float) * fc_context.m_batchsize * fc2.inputs));
        checkCudaErrors(cudaMalloc(&d_dfc2_t,     sizeof(float) * fc_context.m_batchsize * fc1.outputs));
        checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float) * fc_context.m_batchsize * fc2.outputs));
        checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * fc_context.m_batchsize * fc2.outputs));
        // checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM));
        // checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM));

        // Temporary buffers and workspaces
        float *d_onevec;
        checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float) * fc_context.m_batchsize));

        checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));

        // printf("Allocating device variables in proc_id: %d and gpu_id: %d\n", proc_id, gpu_rank);
        float *d_pool2_fc, *d_labels_fc;
        // float *d_pool2_all, *d_pool2_fraction_1, *d_pool2_fraction_2;
        checkCudaErrors(cudaMalloc(&d_pool2_fc,   sizeof(float) * fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM));
        // checkCudaErrors(cudaMalloc(&d_pool2_fraction_1,   sizeof(float) * fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM));
        // checkCudaErrors(cudaMalloc(&d_pool2_fraction_2,   sizeof(float) * fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM));
        // printf("Allocated device variables in proc_id: %d and gpu_id: %d\n", proc_id, gpu_rank);
        checkCudaErrors(cudaMalloc(&d_labels_fc,   sizeof(float) * fc_context.m_batchsize));

        float *d_fc1_all, *d_fc1_r;
        checkCudaErrors(cudaMalloc(&d_fc1_all, sizeof(float) * fc_context.m_batchsize * FC1_OUTPUT_DIM));
        checkCudaErrors(cudaMalloc(&d_fc1_r, sizeof(float) * fc_context.m_batchsize * FC1_OUTPUT_DIM));

        float *d_dfc2_all_t, *d_dfc2_r;
        checkCudaErrors(cudaMalloc(&d_dfc2_all_t, sizeof(float) * fc_context.m_batchsize * fc1.outputs * NUM_WORKERS_DATA_PARALLELISM));
        checkCudaErrors(cudaMalloc(&d_dfc2_r, sizeof(float) * fc_context.m_batchsize * FC1_OUTPUT_DIM * NUM_WORKERS_DATA_PARALLELISM));

        float *d_loss, *d_loss_all, *d_loss_sum, *d_loss_local_sum;
        checkCudaErrors(cudaMalloc(&d_loss, sizeof(float) * fc_context.m_batchsize * fc2.outputs));
        checkCudaErrors(cudaMalloc(&d_loss_local_sum, sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_loss_all, sizeof(float) * NUM_WORKERS_DATA_PARALLELISM)); // verify
        checkCudaErrors(cudaMalloc(&d_loss_sum, sizeof(float) * (ITERATIONS / TEST_INTERVAL))); // verify
        float *h_loss_sum = (float *)malloc((int)(ITERATIONS / TEST_INTERVAL) * sizeof(float)); // verify

        // checkCudaErrors(cudaMalloc(&d_fc1_fraction_1, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM_FRACTION));

        float *d_mypredictions, *d_predictions_all, *d_accuracy;
        checkCudaErrors(cudaMalloc(&d_mypredictions, sizeof(float) * 2));
        checkCudaErrors(cudaMalloc(&d_predictions_all, sizeof(float) * 2 * NUM_WORKERS_DATA_PARALLELISM));
        checkCudaErrors(cudaMalloc(&d_accuracy, sizeof(float)));

        // float *d_fc2_all;
        // checkCudaErrors(cudaMalloc(&d_fc2_all, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM));
        // checkCudaErrors(cudaMalloc(&d_fc2_fraction_1, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM_FRACTION));

        // Variables for testing go here
        float *d_fc1_test, *d_fc1relu_test, *d_fc2_test, *d_fc2smax_test;
        checkCudaErrors(cudaMalloc(&d_fc1_test,     sizeof(float) * TEST_SET_SIZE * fc1.outputs));
        checkCudaErrors(cudaMalloc(&d_fc1relu_test, sizeof(float) * TEST_SET_SIZE * fc1.outputs));
        checkCudaErrors(cudaMalloc(&d_fc2_test,     sizeof(float) * TEST_SET_SIZE * fc2.outputs));
        checkCudaErrors(cudaMalloc(&d_fc2smax_test, sizeof(float) * TEST_SET_SIZE * fc2.outputs));
        // checkCudaErrors(cudaMalloc(&f_class_test, sizeof(float)));
        // checkCudaErrors(cudaMemcpyAsync(f_class_test, &my_class, sizeof(float), cudaMemcpyHostToDevice));

        float *d_pool2_fc_test, *d_labels_fc_test;
        checkCudaErrors(cudaMalloc(&d_pool2_fc_test,   sizeof(float) * TEST_SET_SIZE * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM));
        checkCudaErrors(cudaMalloc(&d_labels_fc_test,   sizeof(float) * TEST_SET_SIZE));
        float *d_fc1_all_test;
        checkCudaErrors(cudaMalloc(&d_fc1_all_test, sizeof(float) * TEST_SET_SIZE * FC1_OUTPUT_DIM));
        float *d_onevec_test;
        checkCudaErrors(cudaMalloc(&d_onevec_test, sizeof(float) * TEST_SET_SIZE));

        // Fill one-vector with ones
        FillOnes<<<RoundUp(fc_context.m_batchsize, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_onevec, fc_context.m_batchsize);

        FillOnes<<<RoundUp(TEST_SET_SIZE, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_onevec_test, TEST_SET_SIZE);
        // Use SGD to train the network
        checkCudaErrors(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERATIONS; ++iter){

            // if((iter + 1) % TEST_INTERVAL == 0)
            // {
            //     SetScalarVal<<<1, 1>>>(d_accuracy, 0.0);

            //     // MPI_Request req_conv_activations[NUM_WORKERS_DATA_PARALLELISM];
            //     // MPI_Request req_labels[NUM_WORKERS_DATA_PARALLELISM];
            //     // for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++ ){
            //         // len = fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM;
            //         // MPI_Recv(d_pool2_fc + i * len, len, MPI_FLOAT, i, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //         // MPI_Recv(d_labels_fc + i * fc_context.m_batchsize, MPI_FLOAT, i, FORPROP_LABELS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //     MPI_Request req_conv_activations;
            //     MPI_Request req_labels;
            //     len = TEST_SET_SIZE * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM;    
            //     int addr = 0;
            //     MPI_Irecv(d_pool2_fc_test, len, MPI_FLOAT, addr, FORPROP_TAG, MPI_COMM_WORLD, &req_conv_activations);
            //     MPI_Irecv(d_labels_fc_test, TEST_SET_SIZE, MPI_FLOAT, addr, FORPROP_LABELS_TAG, MPI_COMM_WORLD, &req_labels);
            //     // }
            //     // fc_context.ForwardPropagationFullyConnectedLayer(d_pool2_all, d_fc1, d_fc1relu, d_pfc1, d_pfc1bias, d_onevec);
            //     // MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req_conv_activations, MPI_STATUSES_IGNORE);
            //     MPI_Wait(&req_conv_activations, MPI_STATUSES_IGNORE);
            //     fc_context.ForwardPropagationFullyConnectedLayer(d_pool2_fc_test, d_fc1_test, d_fc1relu_test, d_pfc1, d_pfc1bias, d_onevec_test);

            //     MPI_Request req[NUM_WORKERS_DATA_PARALLELISM - 1];
            //     count_req = 0;
            //     for( int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
            //         // len = fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * fc1.outputs;
            //         len = TEST_SET_SIZE * fc1.outputs;
            //         if(proc_id == i + NUM_WORKERS_DATA_PARALLELISM) continue;
            //         MPI_Isend(d_fc1relu_test, len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, FORPROP_TAG, MPI_COMM_WORLD, &req[count_req]);
            //         count_req++;
            //     }
            //     MPI_Request req_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
            //     count_req = 0;
            //     for( int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
            //         // len  = fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * fc1.outputs;
            //         len = TEST_SET_SIZE * fc1.outputs;
            //         if( proc_id == i + NUM_WORKERS_DATA_PARALLELISM ) copyData<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_fc1relu_test, d_fc1_all_test + i * len, len); // call copy kernel
            //         // MPI_Recv(d_fc1_all + i * len, len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //         else
            //         {
            //             MPI_Irecv(d_fc1_all_test + i * len, len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, FORPROP_TAG, MPI_COMM_WORLD, &req_recv[count_req]);
            //             count_req++;
            //         }
            //     }
            //     MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req, MPI_STATUSES_IGNORE);
            //     MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_recv, MPI_STATUSES_IGNORE);
            //     fc_context.ForwardPropagationLogisticLayer(d_fc1_all_test, d_fc2_test, d_fc2smax_test, d_pfc2, d_pfc2bias, d_onevec_test);
            //     // MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req_labels, MPI_STATUSES_IGNORE);
            //     MPI_Wait(&req_labels, MPI_STATUS_IGNORE);
            //     // printf("proc_id: %d ===> completed forward prop\n", proc_id);

            //     for(int test_idx = 0; i < TEST_SET_SIZE; i++){
            //         ArgMax<<<1, 1>>>(d_fc2smax_test, d_mypredictions, my_class_size, my_class, i);
            //         // printf("testiteration: %d --- proc_id: %d ===> computed d_mypredictions\n", i, proc_id);
            //         if(proc_id != NUM_WORKERS_DATA_PARALLELISM)
            //         {   
            //             MPI_Request label_send;
            //             MPI_Isend(d_mypredictions, 2, MPI_FLOAT, NUM_WORKERS_DATA_PARALLELISM, SEND_PREDICTIONS_TAG, MPI_COMM_WORLD, &label_send);
            //             MPI_Wait(&label_send, MPI_STATUS_IGNORE);
            //             // printf("testiteration: %d --- proc_id: %d ===> sent my predictions\n", i, proc_id);
            //         }
            //         else
            //         {
            //             MPI_Request label_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
            //             count_req = 0;
            //             for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++)
            //             {   
            //                 if(proc_id == i + NUM_WORKERS_DATA_PARALLELISM)
            //                 {
            //                     copyData<<<1, 2>>>(d_mypredictions, d_predictions_all + i, 2);
            //                 }
            //                 else
            //                 {
            //                     MPI_Irecv(d_predictions_all + i * 2, 2, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, SEND_PREDICTIONS_TAG, MPI_COMM_WORLD, &label_recv[count_req]);
            //                     count_req++;
            //                 }
            //             }
            //             MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, label_recv, MPI_STATUSES_IGNORE);
            //         }
            //         if(proc_id == NUM_WORKERS_DATA_PARALLELISM)
            //         {
            //             ComputeZeroOneLoss<<<1, 1>>>(d_predictions_all, d_labels_fc_test + test_idx, d_accuracy, NUM_WORKERS_DATA_PARALLELISM);
            //             // printf("testiteration: %d --- proc_id: %d ===> computed zerooneloss\n", i, proc_id);
            //         }
            //         // for(int j = 0; j < NUM_WORKERS_DATA_PARALLELISM; j++){
            //         //     if(proc_id == j + NUM_WORKERS_DATA_PARALLELISM){
            //         //         copyData<<<>>>(d_fc2smax + i * my_class_size, d_predictions + j, my_class_size);
            //         //     }
            //         //     else{
            //         //         MPI_Irecv(d_predictions + );
            //         //     }
            //         // }
            //     }

            //     // compute the accuracy here
            //     // ComputeAccuracy<<<1, 1>>>(d_fc2smax, d_labels_fc, test_set_accuracy, TEST_SET_SIZE);
            //     // printf("Iteration #%d ====> Test set accuracy: %f\n", iter, d_accuracy[0]);
            //     if(proc_id == NUM_WORKERS_DATA_PARALLELISM)
            //         PrintVal<<<1, 1>>>(d_accuracy, iter);
            // }

            MPI_Request req_conv_activations[NUM_WORKERS_DATA_PARALLELISM];
            MPI_Request req_labels[NUM_WORKERS_DATA_PARALLELISM];
            for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++ ){
                len = (fc_context.m_batchsize / NUM_WORKERS_DATA_PARALLELISM) * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM;
                // MPI_Recv(d_pool2_fc + i * len, len, MPI_FLOAT, i, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // MPI_Recv(d_labels_fc + i * fc_context.m_batchsize, MPI_FLOAT, i, FORPROP_LABELS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Irecv(d_pool2_fc + i * len, len, MPI_FLOAT, i, FORPROP_TAG, MPI_COMM_WORLD, &req_conv_activations[i]);
                MPI_Irecv(d_labels_fc + i * (fc_context.m_batchsize / NUM_WORKERS_DATA_PARALLELISM), fc_context.m_batchsize / NUM_WORKERS_DATA_PARALLELISM, MPI_FLOAT, i, FORPROP_LABELS_TAG, MPI_COMM_WORLD, &req_labels[i]);
            }
            // fc_context.ForwardPropagationFullyConnectedLayer(d_pool2_all, d_fc1, d_fc1relu, d_pfc1, d_pfc1bias, d_onevec);
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req_conv_activations, MPI_STATUSES_IGNORE); // Verify
            fc_context.ForwardPropagationFullyConnectedLayer(d_pool2_fc, d_fc1, d_fc1relu, d_pfc1, d_pfc1bias, d_onevec);

            MPI_Request req[NUM_WORKERS_DATA_PARALLELISM - 1];
            count_req = 0;
            for( int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = fc_context.m_batchsize * fc1.outputs;
                if(proc_id == i + NUM_WORKERS_DATA_PARALLELISM) continue;
                MPI_Isend(d_fc1relu, len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, FORPROP_FC1_TAG, MPI_COMM_WORLD, &req[count_req]);
                count_req++;
            }
            MPI_Request req_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
            count_req = 0;
            for( int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len  = fc_context.m_batchsize * fc1.outputs;
                if( proc_id == i + NUM_WORKERS_DATA_PARALLELISM ) copyData<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_fc1relu, d_fc1_r + i * len, len); // call copy kernel
                // MPI_Recv(d_fc1_all + i * len, len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                else
                {
                    MPI_Irecv(d_fc1_r + i * len, len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, FORPROP_FC1_TAG, MPI_COMM_WORLD, &req_recv[count_req]);
                    count_req++;
                }
            }
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req, MPI_STATUSES_IGNORE);
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, req_recv, MPI_STATUSES_IGNORE);
            // TODO: Kernel for copying incorrectly ordered FC2 input to d_fc1_all
            len = fc_context.m_batchsize * fc1.outputs * NUM_WORKERS_DATA_PARALLELISM;
            Correct_order<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_fc1_r, d_fc1_all, len, fc_context.m_batchsize, fc1.outputs, NUM_WORKERS_DATA_PARALLELISM);
            fc_context.ForwardPropagationLogisticLayer(d_fc1_all, d_fc2, d_fc2smax, d_pfc2, d_pfc2bias, d_onevec);

            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req_labels, MPI_STATUSES_IGNORE);
            // Compute loss (we need this to generate the training loss curve) using this d_fc2smax variable;
            // these have the predictions of the labels for given input images.

            //const float *labels, const float *predictions, float *loss, int start_label, int my_class_size, int batch_size
            ComputeLoss<<<1, fc_context.m_batchsize * my_class_size>>>(d_labels_fc, d_fc2smax, d_loss, my_class, my_class_size, fc_context.m_batchsize);
            ReduceLoss<<<1, 1>>>(d_loss, d_loss_local_sum, fc_context.m_batchsize * fc2.outputs); // Correct this: Use a good reduce kernel
            
            // PrintVec<<<1, 1>>>(d_loss, fc_context.m_batchsize);
            // PrintVal<<<1, 1>>>(d_loss_local_sum);

            // int start_label, float *labels, float *fc1relu, float *fc2, float *fc2smax, float *dloss_data,
            // float *pfc2, float *pfc2bias, float *gfc2, float *gfc2bias,
            // float *dfc2, float *onevec
            // printf("fc proc_id: %d ====> backprop started\n", proc_id);
            fc_context.BackpropagationLogisticLayer((int)my_class, d_labels_fc, d_fc1_all, d_fc2, d_fc2smax, d_dlossdata, d_pfc2, d_pfc2bias,
                                                    d_gfc2, d_gfc2bias, d_dfc2, d_onevec);
            // printf("fc proc_id: %d ====> backprop done\n", proc_id);

            //...
            MPI_Request fc1_send[NUM_WORKERS_DATA_PARALLELISM - 1];
            count_req = 0;
            for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++)
            {
                len = fc_context.m_batchsize * fc2.inputs;
                if(proc_id == i + NUM_WORKERS_DATA_PARALLELISM) continue;
                MPI_Isend(d_dfc2, len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, BACKPROP_FC_TAG, MPI_COMM_WORLD, &fc1_send[count_req]);
                count_req++;
            }
            MPI_Request fc1_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
            count_req = 0;
            for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = fc_context.m_batchsize * fc2.inputs;
                if(proc_id == i + NUM_WORKERS_DATA_PARALLELISM){
                    copyData<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_dfc2, d_dfc2_r + (i * len), len);
                }
                else
                {
                   MPI_Irecv(d_dfc2_r + (i * len), len, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, BACKPROP_FC_TAG, MPI_COMM_WORLD, &fc1_recv[count_req]);
                   count_req++;
                }
            }
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, fc1_send, MPI_STATUSES_IGNORE);
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, fc1_recv, MPI_STATUSES_IGNORE);
            // TODO: Different Kernel for copying incorrectly ordered d_dfc2_r to d_dfc2_all_t
            len = fc_context.m_batchsize * FC1_OUTPUT_DIM * NUM_WORKERS_DATA_PARALLELISM;
            der_correct_order<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_dfc2_r, d_dfc2_all_t, len, fc_context.m_batchsize, fc1.outputs, NUM_WORKERS_DATA_PARALLELISM, proc_id - NUM_WORKERS_DATA_PARALLELISM);
            len = fc_context.m_batchsize * fc1.outputs;
            ReduceGradients<<<RoundUp(len, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_dfc2_all_t, d_dfc2_t, len, NUM_WORKERS_DATA_PARALLELISM);

            // float *pool2, float *fc1, float *fc1relu, float *dfc1, float*dfc1relu, float *dfc2,
            // float *pfc1, float *pfc1bias, float *gfc1, float *gfc1bias, float *onevec
            // printf("fc proc_id: %d ====> 2nd backprop started\n", proc_id);
            fc_context.BackpropagationFullyConnectedLayer(d_pool2_fc, d_fc1, d_fc1relu, d_dfc1, d_dfc1relu, d_dfc2_t,
                                                          d_pfc1, d_pfc1bias, d_gfc1, d_gfc1bias, d_onevec);
            // printf("fc proc_id: %d ====> 2nd backprop done\n", proc_id);

            MPI_Request req1[NUM_WORKERS_DATA_PARALLELISM];
            // MPI_Status stat1[NUM_WORKERS_DATA_PARALLELISM];
            for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                len = (fc_context.m_batchsize / NUM_WORKERS_DATA_PARALLELISM) * fc1.inputs;
                // printf("fc1.inputs: %d; batch_size: %d; CONV2_OUTPUT_DIM: %d; CONV2_FILTERS: %d\n", fc1.inputs, fc_context.m_batchsize, CONV2_OUTPUT_DIM, CONV2_FILTERS);
                MPI_Isend(d_dfc1 + (i * len), len, MPI_FLOAT, i, BACKPROP_TAG, MPI_COMM_WORLD, &req1[i]);
                // printf("fc proc_id: %d ====> IRecv started to address: %d\n", proc_id, i);
            }
            // float step_size,
            //  float *pfc1, float *pfc1bias,
            //  float *pfc2, float *pfc2bias,
            //  float *gfc1, float *gfc1bias,
            //  float *gfc2, float *gfc2bias
            // printf("fc proc_id: %d ====> update weight started\n", proc_id);
            fc_context.UpdateWeights(STEP_SIZE, 1, d_pfc1, d_pfc1bias, d_pfc2, d_pfc2bias, d_gfc1, d_gfc1bias, d_gfc2, d_gfc2bias);
            // printf("fc proc_id: %d ====> update weight done\n", proc_id);

            // send your loss function to the NUM_DATA_PARALLELISM thread
            if(proc_id != NUM_WORKERS_DATA_PARALLELISM){
                MPI_Request loss_send;
                MPI_Isend(d_loss_local_sum, 1, MPI_FLOAT, NUM_WORKERS_DATA_PARALLELISM, TRAINING_LOSS_TAG, MPI_COMM_WORLD, &loss_send);
                MPI_Wait(&loss_send, MPI_STATUS_IGNORE);
            }
            // if(proc_id != NUM_WORKERS_DATA_PARALLELISM){
            //     MPI_Request loss_send[NUM_WORKERS_DATA_PARALLELISM - 1];
            //     count_req = 0;
            //     for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
            //         if(proc_id == i + NUM_WORKERS_DATA_PARALLELISM)
            //             continue;
            //         else  
            //         {
            //             MPI_Isend(d_loss_local_sum, 1, MPI_FLOAT, NUM_WORKERS_DATA_PARALLELISM, TRAINING_LOSS_TAG, MPI_COMM_WORLD, &loss_send[count_req]);
            //             count_req++;
            //         }
            //     }
            //     MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, loss_send, MPI_STATUSES_IGNORE);
            // }

            if(proc_id == NUM_WORKERS_DATA_PARALLELISM){
                MPI_Request loss_recv[NUM_WORKERS_DATA_PARALLELISM - 1];
                count_req = 0;
                for(int i = 0; i < NUM_WORKERS_DATA_PARALLELISM; i++){
                    len = fc_context.m_batchsize;
                    if(proc_id == i + NUM_WORKERS_DATA_PARALLELISM)
                        // d_loss_all[i] = d_loss;
                        copyData<<<1, 1>>>(d_loss_local_sum, d_loss_all + i, 1);
                    else
                    {
                        MPI_Irecv(d_loss_all + i, 1, MPI_FLOAT, i + NUM_WORKERS_DATA_PARALLELISM, TRAINING_LOSS_TAG, MPI_COMM_WORLD, &loss_recv[count_req]);
                        count_req++;
                    }
                }
                MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM - 1, loss_recv, MPI_STATUSES_IGNORE);
                // ReduceLoss<<<1, 1>>>(d_loss_all, d_loss_sum, (int)((iter + 1) / TEST_INTERVAL), NUM_WORKERS_DATA_PARALLELISM);
                ReduceLossToPlotVariable<<<1, 1>>>(d_loss_all, d_loss_sum, (int)((iter + 1) / TEST_INTERVAL), NUM_WORKERS_DATA_PARALLELISM);
                // printf("loss: %f\n", d_loss_sum);
                // PrintVal<<<1, 1>>>(d_loss_sum, iter);
            }

            // printf("fc proc_id: %d ====> send waiting\n", proc_id);
            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req1, MPI_STATUSES_IGNORE);
            // printf("fc proc_id: %d ====> send waited\n", proc_id);
        }
  
      if(proc_id == NUM_WORKERS_DATA_PARALLELISM)
      {
          checkCudaErrors(cudaMemcpy(h_loss_sum, &d_loss_sum[0], sizeof(float) * (int)(ITERATIONS / TEST_INTERVAL), cudaMemcpyDeviceToHost));
          std::ofstream myfile ("training_loss.txt");
          if (myfile.is_open())
          {
              for(int i = 0; i < (int)(ITERATIONS / TEST_INTERVAL); i++)
              {
                  myfile << h_loss_sum[i] << "\n";
              }
              myfile.close();
          }
      }
  
    }
    MPI_Finalize();
    return 0;
}
