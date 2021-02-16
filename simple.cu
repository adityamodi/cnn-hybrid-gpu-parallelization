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
#define BLOCK_WIDTH 128
#define ITERATIONS 1000
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

#define RANDOM_SEED -1

#define CONV_ADDRESS_1 0
#define CONV_ADDRESS_2 1
#define FC_ADDRESS_1 2
#define FC_ADDRESS_2 3

#define FC1_OUTPUT_DIM 500
#define FC1_OUTPUT_DIM_FRACTION 250
#define FC2_OUTPUT_DIM 10
#define FC2_OUTPUT_DIM_FRACTION 5

__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0f;
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
        out_height = out_width - filter_size_ + 1;
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
                         float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
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

    void UpdateWeights(float step_size, ConvBiasLayer& conv1, ConvBiasLayer& conv2, float *pconv1, float *pconv1bias,
                      float *pconv2, float *pconv2bias, float *gconv1, float *gconv1bias, float *gconv2, float *gconv2bias)
    {
          float alpha = -step_size;
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

    void ForwardPropagationPreSoftmaxLayer(float *fc1relu, float *fc2, float *pfc2, float *pfc2bias, float *onevec)
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
    }

    void ForwardPropagationSoftmax(float *fc2, float *result)
    {
        float alpha = 1.0f, beta = 0.0f;
        // Softmax loss
        checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                       &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
    }

    void BackpropSoftmax()
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
    // printf("proc_id %d linked to device %d\n", proc_id, gpu_rank);

    if(proc_id < NUM_WORKERS_DATA_PARALLELISM)
    {
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

        printf("Training...\n");

        // Use SGD to train the network
        checkCudaErrors(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        for(int iter = 0; iter < ITERATIONS; ++iter)
        {
            checkCudaErrors(cudaSetDevice(gpu_rank));
            // Train
            // printf("iteration #%d\n", iter);
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
            MPI_Status stat[NUM_WORKERS_DATA_PARALLELISM];

            MPI_Isend(d_pool2, conv_context.m_batchsize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride), MPI_FLOAT, FC_ADDRESS_1, FORPROP_TAG, MPI_COMM_WORLD, &req[0]);
            MPI_Isend(d_pool2, conv_context.m_batchsize * conv2.out_channels * (conv2.out_height / pool2.stride) * (conv2.out_width / pool2.stride), MPI_FLOAT, FC_ADDRESS_2, FORPROP_TAG, MPI_COMM_WORLD, &req[1]);

            MPI_Waitall(NUM_WORKERS_DATA_PARALLELISM, req, stat);
        }
    }
    if(proc_id >= NUM_WORKERS_DATA_PARALLELISM && proc_id < num_processors)
    {
        // printf("proc_id: %d\n", proc_id);
        checkCudaErrors(cudaSetDevice(proc_id));
        // Check the dimensions
        FullyConnectedLayer fc1(CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM * CONV2_FILTERS, FC1_OUTPUT_DIM_FRACTION);
        FullyConnectedLayer fc2(FC1_OUTPUT_DIM, FC2_OUTPUT_DIM_FRACTION);

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

        float *d_fc1, *d_fc1relu, *d_fc2, *d_fc2smax;
        checkCudaErrors(cudaMalloc(&d_fc1,     sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * fc1.outputs));
        checkCudaErrors(cudaMalloc(&d_fc1relu, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * fc1.outputs));
        checkCudaErrors(cudaMalloc(&d_fc2,     sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * fc2.outputs));
        // checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float) * fc_context.m_batchsize * fc2.outputs));
        checkCudaErrors(cudaMalloc(&d_fc2smax, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM));

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

        float *d_dfc1, *d_dfc1relu, *d_dfc2, *d_dfc2smax, *d_dlossdata;
        checkCudaErrors(cudaMalloc(&d_dfc1,     sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * fc1.inputs));
        checkCudaErrors(cudaMalloc(&d_dfc1relu, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * fc1.outputs));
        checkCudaErrors(cudaMalloc(&d_dfc2,     sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * fc2.inputs));
        // checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float) * fc_context.m_batchsize * fc2.outputs));
        // checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * fc_context.m_batchsize * fc2.outputs));
        checkCudaErrors(cudaMalloc(&d_dfc2smax, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM));
        checkCudaErrors(cudaMalloc(&d_dlossdata,sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM));

        // Temporary buffers and workspaces
        float *d_onevec;
        checkCudaErrors(cudaMalloc(&d_onevec, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM));

        checkCudaErrors(cudaMemcpyAsync(d_pfc1, &fc1.pneurons[0],      sizeof(float) * fc1.pneurons.size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pfc1bias, &fc1.pbias[0],     sizeof(float) * fc1.pbias.size(),    cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pfc2, &fc2.pneurons[0],      sizeof(float) * fc2.pneurons.size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_pfc2bias, &fc2.pbias[0],     sizeof(float) * fc2.pbias.size(),    cudaMemcpyHostToDevice));

        // printf("Allocating device variables in proc_id: %d and gpu_id: %d\n", proc_id, gpu_rank);
        float *d_pool2_all, *d_pool2_fraction_1, *d_pool2_fraction_2;
        checkCudaErrors(cudaMalloc(&d_pool2_all,   sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM));
        checkCudaErrors(cudaMalloc(&d_pool2_fraction_1,   sizeof(float) * fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM));
        checkCudaErrors(cudaMalloc(&d_pool2_fraction_2,   sizeof(float) * fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM));
        // printf("Allocated device variables in proc_id: %d and gpu_id: %d\n", proc_id, gpu_rank);

        float *d_fc1_all, *d_fc1_fraction_1;
        checkCudaErrors(cudaMalloc(&d_fc1_all, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM));
        checkCudaErrors(cudaMalloc(&d_fc1_fraction_1, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM_FRACTION));

        float *d_fc2_all, *d_fc2_fraction_1;
        checkCudaErrors(cudaMalloc(&d_fc2_all, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM));
        checkCudaErrors(cudaMalloc(&d_fc2_fraction_1, sizeof(float) * fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM_FRACTION));

        // Fill one-vector with ones
        FillOnes<<<RoundUp(fc_context.m_batchsize, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_onevec, fc_context.m_batchsize);
        // Use SGD to train the network
        checkCudaErrors(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < ITERATIONS; ++iter)
        {
            MPI_Recv(d_pool2_fraction_1, fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM, MPI_FLOAT,
                     CONV_ADDRESS_1, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(d_pool2_fraction_2, fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM, MPI_FLOAT,
                     CONV_ADDRESS_2, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MergeMatrices<<<RoundUp(fc_context.m_batchsize, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_pool2_fraction_1, d_pool2_fraction_2, d_pool2_all, fc_context.m_batchsize * CONV2_FILTERS * CONV2_OUTPUT_DIM * CONV2_OUTPUT_DIM);
            fc_context.ForwardPropagationFullyConnectedLayer(d_pool2_all, d_fc1, d_fc1relu, d_pfc1, d_pfc1bias, d_onevec);

            MPI_Request req;
            if(proc_id == FC_ADDRESS_1)
            {
                MPI_Isend(d_fc1relu, fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM_FRACTION, MPI_FLOAT, FC_ADDRESS_2, FORPROP_TAG, MPI_COMM_WORLD, &req);
                MPI_Recv(d_fc1_fraction_1, fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM_FRACTION, MPI_FLOAT, FC_ADDRESS_2, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else if(proc_id == FC_ADDRESS_2)
            {
                MPI_Isend(d_fc1relu, fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM_FRACTION, MPI_FLOAT, FC_ADDRESS_1, FORPROP_TAG, MPI_COMM_WORLD, &req);
                MPI_Recv(d_fc1_fraction_1, fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC1_OUTPUT_DIM_FRACTION, MPI_FLOAT, FC_ADDRESS_1, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // this needs to be a different merging routine
            MergeMatrices<<<RoundUp(fc_context.m_batchsize, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_fc1relu, d_fc1_fraction_1, d_fc1_all, fc_context.m_batchsize * FC1_OUTPUT_DIM);
            fc_context.ForwardPropagationPreSoftmaxLayer(d_fc1_all, d_fc2, d_pfc2, d_pfc2bias, d_onevec);

            if(proc_id == FC_ADDRESS_2)
            {
                MPI_Request req;
                MPI_Isend(d_fc2, fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM_FRACTION, MPI_FLOAT, FC_ADDRESS_1, FORPROP_TAG, MPI_COMM_WORLD, &req);
            }
            else if(proc_id == FC_ADDRESS_1)
            {
                MPI_Recv(d_fc2_fraction_1, fc_context.m_batchsize * NUM_WORKERS_DATA_PARALLELISM * FC2_OUTPUT_DIM_FRACTION, MPI_FLOAT, FC_ADDRESS_2, FORPROP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // this needs to be a different merging routine
                MergeMatrices<<<RoundUp(fc_context.m_batchsize, BLOCK_WIDTH), BLOCK_WIDTH>>>(d_fc2, d_fc2_fraction_1, d_fc2_all, fc_context.m_batchsize * FC2_OUTPUT_DIM_FRACTION);
                fc_context.ForwardPropagationSoftmax(d_fc2_all, d_fc2smax);
            }
        }
    }
    MPI_Finalize();
    return 0;
}
