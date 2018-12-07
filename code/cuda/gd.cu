#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

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

bool withinAlphaMSE(float lower, float upper, estimate_t* est,
                        float* X, float* Y, int N) {
    float est_MSE = 0.0;
    for (int i = 0; i < N; i++) {
        float error = Y[i] - ((est->b1) * X[i] + est->b0);
        est_MSE += error * error;
    }
    est_MSE /= static_cast<float>(N);

    return lower < est_MSE && est_MSE < upper;
}

__device__ __inline__ float
evaluateCuda(estimate_t* estimate, float x){
  return (estimate->b1)*x + estimate->b0;
}

__device__ __inline__ float
getdB0Cuda(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluateCuda(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction);
}

__device__ __inline__ float
getdB1Cuda(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluateCuda(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction)*x;
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
sgd_step(int N, float* device_X, float* device_Y,
               estimate_t* device_estimates, curandState* states, int* times) {
  clock_t start = clock();
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  curandState localState = states[index];
  int pi = curand(&localState) % N;
  states[index] = localState;

  float db0 = getdB0Cuda(device_X[pi], device_Y[pi], device_estimates+index, N);
  float db1 = getdB1Cuda(device_X[pi], device_Y[pi], device_estimates+index, N);

  device_estimates[index].b0 -= (STEP_SIZE_STOCH * db0);
  device_estimates[index].b1 -= (STEP_SIZE_STOCH * db1);

  clock_t end = clock();
  times[index] = (int)(end - start);
}

// Running SGD with all threads each sampling one point and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdCuda(int N, float* x, float* y, float alpha, float opt,
                    int blocks, int threadsPerBlock){

  int* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  int totalThreads = blocks * threadsPerBlock;

  cudaMalloc((void **)&device_times, totalThreads * sizeof(int));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * totalThreads);

  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  for (int i = 0; i < totalThreads; i++) {
      estimates[i].b0 = INIT_B0;
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
  int times[totalThreads];
  int total_time = 0;
  while(true)
  {
    sgd_step<<<blocks, threadsPerBlock>>>(N, device_X, device_Y, device_estimates, states, device_times);
    cudaThreadSynchronize();

    cudaMemcpy(times, device_times, totalThreads * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(estimates, device_estimates, totalThreads * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);

    int max_time = 0;
    for (int i = 0; i < totalThreads; i++) {
        if (times[i] > max_time) max_time = times[i];
    }
    total_time += max_time;

    ret -> b0 = 0.0;
    ret -> b1 = 0.0;
    for(int j = 0; j < totalThreads; j++) {
      ret -> b0 += estimates[j].b0 / static_cast<float>(totalThreads);
      ret -> b1 += estimates[j].b1 / static_cast<float>(totalThreads);
    }

    if(num_steps > ITER_LIMIT || withinAlphaMSE(lower, upper, ret, x, y, N)) {
        break;
    }

    num_steps += 1;
  }

  printf("Num iterations: %d\n", num_steps);
  printf("Num of clock cycles: %d\n", total_time);

  return ret;
}

//-----------------------------------------------------------------------------

// Assumes the number of indexes is equal to N
__global__ void
sgdStepByBlock(int N, float* device_X, float* device_Y,
               estimate_t* device_estimates, int k, curandState* states, int* times) {
  clock_t start = clock();
  __shared__ estimate_t thread_estimate[THREADS_PER_BLOCK];

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  thread_estimate[threadIdx.x].b0 = 0.0;
  thread_estimate[threadIdx.x].b1 = 0.0;
  curandState localState = states[index];

  for(int i = 0; i < k; i++)
  {
      int pi = curand(&localState) % N;
      thread_estimate[threadIdx.x].b0 +=
        (getdB0Cuda(device_X[pi], device_Y[pi], device_estimates+blockIdx.x, N) /
        static_cast<float>(k));
      thread_estimate[threadIdx.x].b1 +=
        (getdB1Cuda(device_X[pi], device_Y[pi], device_estimates+blockIdx.x, N) /
        static_cast<float>(k));
  }
  states[index] = localState;

  __syncthreads();

  if(threadIdx.x == 0)
  {
      float db0 = 0.0;
      float db1 = 0.0;
      for(int i = 0; i < blockDim.x; i++){
          db0 += thread_estimate[i].b0 / static_cast<float>(blockDim.x);
          db1 += thread_estimate[i].b1 / static_cast<float>(blockDim.x);
      }

      device_estimates[blockIdx.x].b0 -= (STEP_SIZE_STOCH * db0);
      device_estimates[blockIdx.x].b1 -= (STEP_SIZE_STOCH * db1);
  }
  clock_t stop = clock();
  times[index] = (int)(stop - start);
}

// Running SGD on each block with all threads sampling k points and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdCudaByBlock(int N, float* x, float* y, float alpha, float opt,
                     int k, int blocks, int threadsPerBlock){
  int* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  int totalThreads = blocks * threadsPerBlock;

  cudaMalloc((void **)&device_times, totalThreads * sizeof(int));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * blocks);

  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  for (int i = 0; i < totalThreads; i++) {
      estimates[i].b0 = INIT_B0;
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

  int times[totalThreads];
  int num_steps = 0;
  int total_time = 0;

  while(true)
  {
    sgdStepByBlock<<<blocks, threadsPerBlock>>>(N, device_X, device_Y, device_estimates, k, states, device_times);
    cudaThreadSynchronize();

    cudaMemcpy(times, device_times, totalThreads * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(estimates, device_estimates, blocks * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);

    int max_time = 0;
    for (int i = 0; i < totalThreads; i++) {
        if (times[i] > max_time) max_time = times[i];
    }
    total_time += max_time;

    ret -> b0 = 0.0;
    ret -> b1 = 0.0;
    for(int j = 0; j < blocks; j++) {
      ret -> b0 += estimates[j].b0 / static_cast<float>(blocks);
      ret -> b1 += estimates[j].b1 / static_cast<float>(blocks);
    }

    if(num_steps > ITER_LIMIT || withinAlphaMSE(lower, upper, ret, x, y, N)) {
        break;
    }

    num_steps += 1;
  }

  printf("Num iterations: %d\n", num_steps);
  printf("Num of clock cycles: %d\n", total_time);

  return ret;
}
