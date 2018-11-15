#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "regression.h"

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

__device__ __inline__ float
evaluateCuda(estimate_t* estimate, float x){
  return (estimate->b1)*x + (estimate->b0);
}

__device__ __inline__ float
getdB0Cuda(float x, float y, estimate_t* estimate){
  float prediction = evaluateCuda(estimate, x);
  return -2.0 * (y-prediction);
}

__device__ __inline__ float
getdB1Cuda(float x, float y, estimate_t* estimate){
  float prediction = evaluateCuda(estimate, x);
  return -2.0 * (y-prediction)*x;
}

// Assumes the number of indexes is equal to N
__global__ void
descend(int N, float* device_X, float* device_Y, float* device_result) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float db0All;
  __shared__ float db1All;
  __shared__ estimate_t estimate;

  if(index == 0){
    db0All = 0.0;
    db1All = 0.0;
    estimate.b0 = INIT_B0;
    estimate.b1 = INIT_B1;
  }

  __syncthreads();

  for(int i = 0; i < NUM_ITER; i++)
  {
    float db0 = (1.0 / static_cast<float>(N)) * getdB0Cuda(device_X[index], device_Y[index], &estimate);
    float db1 = (1.0 / static_cast<float>(N)) * getdB1Cuda(device_X[index], device_Y[index], &estimate);

    atomicAdd(&db0All, db0);
    atomicAdd(&db1All, db1);

    __syncthreads();

    if(i < 5 && index == 0){
      printf("cuda: %f \t %f \n", db0All, db1All);
    }

    if(index == 0){
      estimate.b0 = (estimate.b0) - (STEP_SIZE * db0All);
      estimate.b1 = (estimate.b1) - (STEP_SIZE * db1All);
    }

    __syncthreads();
  }

  if(index == 0){ //root
    device_result[0] = estimate.b0;
    device_result[1] = estimate.b1;
  }
}

estimate_t* bgdCuda(int N, float* x, float* y){

  float* device_X;
  float* device_Y;
  float* device_result;

  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_result, sizeof(float) * 2);

  cudaMemcpy(device_X, x, N * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(device_Y, y, N * sizeof(float),
             cudaMemcpyHostToDevice);

  int blocks = 1;
  int threadsPerBlock = N; //ASSUMING N < 1024
  // int totalThreads = blocks * threadsPerBlock;

  descend<<<blocks, threadsPerBlock>>>(N, device_X, device_Y, device_result);

  float resultArray[2];
  cudaMemcpy(&resultArray, device_result, sizeof(float) * 2,
             cudaMemcpyDeviceToHost);

  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b0 = resultArray[0];
  estimate -> b1 = resultArray[1];

  return estimate;
}


float evaluateCudaCopy(estimate_t* estimate, float x){
  return (estimate->b1)*x + (estimate->b0);
}

float getdB0CudaCopy(float x, float y, estimate_t* estimate){
  float prediction = evaluateCudaCopy(estimate, x);
  return -2.0 * (y-prediction);
}

float getdB1CudaCopy(float x, float y, estimate_t* estimate){
  float prediction = evaluateCudaCopy(estimate, x);
  return -2.0 * (y-prediction)*x;
}

estimate_t* bgdCudaCopy(int N, float* x, float* y)
{

  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b0 = 0.0;
  estimate -> b1 = 0.0;

  //pick a point randomly
  for(int i = 0; i < NUM_ITER; i++){

    float db0 = 0;
    float db1 = 0;
    for(int j = 0; j < N; j++)
    {
      db0 += (1.0 / static_cast<float>(N)) * getdB0CudaCopy(x[j], y[j], estimate);
      db1 += (1.0 / static_cast<float>(N)) * getdB1CudaCopy(x[j], y[j], estimate);
    }
    if(i == 0){
      // printf("%f \t %f \n", db0, db1);
    }
    estimate -> b0 = (estimate -> b0) - (STEP_SIZE * db0);
    estimate -> b1 = (estimate -> b1) - (STEP_SIZE * db1);
  }
  return estimate;
}
