#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>

#include "regression.h"

#define THREADS_PER_BLOCK 16

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


__global__ void
setup_kernel(curandState *states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(418, index, 0, &states[index]);
}

// Assumes the number of indexes is equal to N
__global__ void
sgd_step(int N, float* device_X, float* device_Y, estimate_t* device_estimates, curandState* states) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  curandState localState = states[index];

  int pi = curand(&localState) % N;
  states[index] = localState;

  float db1 = (1.0 / static_cast<float>(N)) * getdB1Cuda(device_X[pi], device_Y[pi],
                                                    device_estimates + index);

  device_estimates[index].b1 -= (STEP_SIZE_STOCH * db1);
}

// Running SGD with all threads each sampling one point and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdCuda(int N, float* x, float* y, float alpha, float opt,
                    int blocks, int threadsPerBlock){

  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  int totalThreads = blocks * threadsPerBlock;

  curandState *states;
  cudaMalloc((void**)&states, totalThreads * sizeof(curandState));

  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * totalThreads);

  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  for (int i = 0; i < totalThreads; i++) {
      estimates[i].b1 = INIT_B1;
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, totalThreads * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);

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
    int flag = 0;
    for(int j = 0; j < blocks; j++) {
      ret -> b1 += estimates[j].b1;
      if ((lower < estimates[j].b1) && (estimates[j].b1 < upper)) {
          ret -> b1 = estimates[j].b1;
          flag = 1;
          break;
      }
    }
    if (flag) break;

    ret -> b1 /= static_cast<float>(totalThreads);

    if(num_steps > ITER_LIMIT || (lower < (ret -> b1) && (ret -> b1) < upper))
      break;

    num_steps += 1;
  }

  printf("Num iterations: %d\n", num_steps);

  return ret;
}

//-----------------------------------------------------------------------------

// Assumes the number of indexes is equal to N
__global__ void
sgdStepByBlock(int N, float* device_X, float* device_Y,
               estimate_t* device_estimates, int k, curandState* states) {

  __shared__ float thread_db1[THREADS_PER_BLOCK];

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  thread_db1[threadIdx.x] = 0.0;
  curandState localState = states[index];

  for(int i = 0; i < k; i++)
  {
      int pi = curand(&localState) % N;
      thread_db1[threadIdx.x] += (1.0 / static_cast<float>(N)) *
            getdB1Cuda(device_X[pi], device_Y[pi], device_estimates+blockIdx.x);
  }
  thread_db1[threadIdx.x] /= static_cast<float>(k);
  states[index] = localState;

  __syncthreads();


  if(threadIdx.x == 0)
  {
      float db1 = 0.0;
      for(int i = 0; i < blockDim.x; i++){
          db1 += thread_db1[i];
      }
      db1 /= static_cast<float>(blockDim.x);

      device_estimates[blockIdx.x].b1 -= (STEP_SIZE_STOCH * db1);
  }

}

// Running SGD on each block with all threads sampling k points and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdCudaByBlock(int N, float* x, float* y, float alpha, float opt,
                     int k, int blocks, int threadsPerBlock){
  curandState *states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  int totalThreads = blocks * threadsPerBlock;

  cudaMalloc((void**)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * blocks);

  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  for (int i = 0; i < totalThreads; i++) {
      estimates[i].b1 = INIT_B1;
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, blocks * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);

  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  float upper = opt + (alpha/2.0)*opt;
  float lower = opt - (alpha/2.0)*opt;

  int num_steps = 0;
  while(true)
  {
    sgdStepByBlock<<<blocks, threadsPerBlock>>>(N, device_X, device_Y, device_estimates, k, states);

    cudaMemcpy(estimates, device_estimates, blocks * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);

    ret -> b1 = 0.0;
    int flag = 0;
    for(int j = 0; j < blocks; j++) {
      ret -> b1 += estimates[j].b1;
      if ((lower < estimates[j].b1) && (estimates[j].b1 < upper)) {
          ret -> b1 = estimates[j].b1;
          flag = 1;
          break;
      }
    }

    if (flag) break;

    ret -> b1 /= static_cast<float>(blocks);

    if(num_steps > ITER_LIMIT || (lower < (ret -> b1) && (ret -> b1) < upper))
      break;

    num_steps += 1;
  }

  printf("Num iterations: %d\n", num_steps);

  return ret;
}
