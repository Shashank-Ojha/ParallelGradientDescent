#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>

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
  return (estimate->b1)*x;
}

__device__ __inline__ float
getdB1Cuda(float x, float y, estimate_t* estimate){
  float prediction = evaluateCuda(estimate, x);
  return -2.0 * (y-prediction)*x;
}

// estimate_t* bgdCuda(int N, float* x, float* y){
//
//   float* device_X;
//   float* device_Y;
//   float* device_result;
//
//   cudaMalloc((void **)&device_X, sizeof(float) * N);
//   cudaMalloc((void **)&device_Y, sizeof(float) * N);
//   cudaMalloc((void **)&device_result, sizeof(float));
//
//   cudaMemcpy(device_X, x, N * sizeof(float),
//              cudaMemcpyHostToDevice);
//
//   cudaMemcpy(device_Y, y, N * sizeof(float),
//              cudaMemcpyHostToDevice);
//
//   int blocks = 1;
//   int threadsPerBlock = N; //ASSUMING N < 1024
//   // int totalThreads = blocks * threadsPerBlock;
//
//   descend<<<blocks, threadsPerBlock>>>(N, device_X, device_Y, device_result);
//
//   float result;
//   cudaMemcpy(&result, device_result, sizeof(float),
//              cudaMemcpyDeviceToHost);
//
//   estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
//   estimate -> b1 = resultArray[1];
//
//   return estimate;
// }

//-----------------------------------------------------------------------------

// Assumes the number of indexes is equal to N
__global__ void
sgd_step(int N, float* device_X, float* device_Y, estimate_t* device_estimates curandState* states) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  int pi = CURAND() * N;

  float db1 = (1.0 / static_cast<float>(N)) * getdB1Cuda(device_X[pi], device_Y[pi],
                                                    device_estimates + index);

  device_estimates[index].b1 -= (STEP_SIZE_STOCH * db1);

}

estimate_t* sgdCuda(int N, float* x, float* y, float alpha, float opt){

  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  curandState *states;
  cudaMalloc((void**)&states, totalThreads * sizeof(curandState));


  int blocks = 8;
  int threadsPerBlock = 128;
  int totalThreads = blocks * threadsPerBlock;

  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * totalThreads);

  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));

  cudaMemcpy(device_X, x, N * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(device_Y, y, N * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(device_estimates, estimates, totalThreads * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  float upper = opt + (alpha/2.0)*opt;
  float lower = opt - (alpha/2.0)*opt;

  int num_steps = 0;
  while(true)
  {
    sgd_step<<<blocks, threadsPerBlock>>>(N, device_X, device_Y, device_estimates, states);

    cudaMemcpy(estimates, device_estimates, totalThreads * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);

    ret -> b1 = 0.0;
    for(int j = 0; j < totalThreads; j++) {
      ret -> b1 += estimates[j].b1;
    }

    ret -> b1 /= static_cast<float>(totalThreads);

    if(num_steps > ITER_LIMIT || (lower < (ret -> b1) && (ret -> b1) < upper))
      break;

    num_steps += 1;
  }

  return ret;
}
