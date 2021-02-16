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

#define BLOCK_WIDTH 128
#define GPU_ID 0
#define NUM_ITERATIONS 1000
#define RANDOM_SEED 1
#define BATCH_SIZE 64
#define STEP_SIZE 0.01


int ceilfn(int x, int y)
{
    return (x + y - 1) / y;
}

struct ConvBiasLayer
{
    int in_channels, out_channels, filter_size;
    int in_width, in_height, out_width, out_height;
    std::vector<float> pconv, pbias;
    ConvBiasLayer(int in_channels_, int out_channels_, int filter_size_,
                  int width_, int height_) : pconv(in_channels_ * filter_size_ * filter_size_ * out_channels_),
                  pbias(out_channels_)
    {
        in_channels = in_channels_;
        out_channels = out_channels_;
        filter_size = filter_size_;
        in_width = width_;
        in_height = height_;
        out_width = width_ - filter_size_ + 1;
        out_height = height_ - filter_size_ + 1;
    }
};

struct MaxPoolLayer
{
    int size, int stride;
    MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};

struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_) : inputs(inputs_), outputs(outputs_),
                        pneurons(inputs_ * outputs_), pbias(outputs_) {}
};

__global__ void FillOnes(float *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;
    vec[idx] = 1.0f;
}

__global__ void SoftmaxLossBackprop(float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batch_size)
        return;
    int label_value = static_cast<int>(label[idx]);
    diff[idx * num_labels + label_value] -= 1.0f;
}

struct TrainingContext
{
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor, pool1Tensor,
                            conv2Tensor, conv2BiasTensor, pool2Tensor, fc1Tensor, fc2Tensor;
    cudnnFilterDescriptor_t conv1filterDesc, conv2filterDesc;
    cudnnConvolutionFwdAlgo_t conv1algo, conv2algo;
    cudnnConvolutionBwdFilterAlgo_t conv1bwfalgo, conv2bwfalgo;
    cudnnConvolutionBwdDataAlgo_t conv2bwdalgo;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnActivationDescriptor_t fc1Activation;

    int m_gpuid;
    int m_batchsize;
    size_t m_workspaceSize;

    FullyConnectedLayer& ref_fc1, ref_fc2;

    TrainingContext& operator=(const TrainingContext&) = delete;
    TrainingContext(const TrainingContext&) = delete;
    TrainingContext(int gpuid, int batch_size, ConvBiasLayer& conv1, MaxPoolLayer& pool1, ConvBiasLayer& conv2, MaxPoolLayer& pool2,
                    FullyConnectedLayer& fc1, FullyConnectedLayer& fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid)
    {
        m_batchsize = batch_size;

        checkCudaErrors(cudaSetDevice(gpuid));
        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCUDNN(cudnnCreate(&cudnnHandle));
        // Create tensor descriptors
        checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv1BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&conv2BiasTensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&pool2Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc1Tensor));
        checkCUDNN(cudnnCreateTensorDescriptor(&fc2Tensor));

        checkCUDNN(cudnnCreateActivationDescriptor(&fc1Activation));

        checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&conv2filterDesc));

        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2Desc));

        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));


        // Set tensor descriptor sizes
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


        // Set convolution tensor sizes and compute workspace size
        size_t workspace = 0;
        workspace = std::max(workspace, SetFwdConvolutionTensors(conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, &conv1bwfalgo, nullptr));

        workspace = std::max(workspace, SetFwdConvolutionTensors(conv2, pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, conv2algo));
        workspace = std::max(workspace, SetBwdConvolutionTensors(pool1Tensor, conv2Tensor, conv2filterDesc, conv2Desc, &conv2bwfalgo, &conv2bwdalgo));

        // The workspace is allocated later (if necessary)
        m_workspaceSize = workspace;
    }

    ~TrainingContext()
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
        checkCUDNN(cudnnDestroyTensorDescriptor(fc1Tensor));
        checkCUDNN(cudnnDestroyTensorDescriptor(fc2Tensor));
        checkCUDNN(cudnnDestroyActivationDescriptor(fc1Activation));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(conv2filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(conv2Desc));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    }

    
};
