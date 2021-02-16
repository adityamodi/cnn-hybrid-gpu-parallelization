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

#define BLOCK_WIDTH 4000000

__global__ void FillJunk(float *vec, int size)
{
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // if(idx >= size)
    //     return;
    // vec[idx] = (float)idx;
    // printf("val generated: %f\n", vec[idx]);
    for(int i = 0; i < size; i++)
        vec[i] = (float)i;
}

__global__ void PrintVec(float *vec, int size)
{
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // if(idx >= size)
    //     return;
    // printf("idx: %f\n", vec[idx]);
    for(int i = 0; i < size; i++)
        printf("idx: %f\n", vec[i]);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int num_processors, proc_id;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processors);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    printf("Processor %d assigned to GPU: %d\n", proc_id, proc_id + 2);

    float *d_svec;
    if(proc_id == 0)
    {
        cudaSetDevice(proc_id + 2);
        cudaMalloc(&d_svec, sizeof(float) * BLOCK_WIDTH);
        FillJunk<<<1, 1>>>(d_svec, BLOCK_WIDTH);
        cudaDeviceSynchronize();
        printf("the vector that is being sent:\n");
        PrintVec<<<1, 1>>>(d_svec, BLOCK_WIDTH);
        MPI_Send(d_svec, BLOCK_WIDTH, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    }

    // printf("the vector that is being sent:\n");
    // for(int i = 0; i < BLOCK_WIDTH; i++)
    // {
    //     printf("%f\n", d_svec[i]);
    // }
    // MPI_Send(d_svec, BLOCK_WIDTH, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    // }
    else if(proc_id == 1)
    {
        cudaSetDevice(proc_id + 2);
        cudaMalloc(&d_svec, sizeof(float) * BLOCK_WIDTH);
        MPI_Recv(d_svec, BLOCK_WIDTH, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("the vector that is received:\n");
        PrintVec<<<1, 1>>>(d_svec, BLOCK_WIDTH);
    }

    MPI_Finalize();
}
