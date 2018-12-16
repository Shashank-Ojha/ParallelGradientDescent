#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>

#include "regression.h"

#define THREADS_PER_BLOCK 8

void printCudaInfo()
{
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

void average_estimates(estimate_t* estimates, estimate_t* ret, int num_threads)
{
  ret->b0 = 0.0;
  ret->b1 = 0.0;
  ret->b2 = 0.0;
  ret->b3 = 0.0;
  for(int i = 0; i < num_threads; i++) {
    ret -> b0 += (estimates+i) -> b0 / static_cast<float>(num_threads);
    ret -> b1 += (estimates+i) -> b1 / static_cast<float>(num_threads);
    ret -> b2 += (estimates+i) -> b2 / static_cast<float>(num_threads);
    ret -> b3 += (estimates+i) -> b3 / static_cast<float>(num_threads);
  }
}

long get_max_time(long* times, int num_threads) {
    long max = 0;
    for(int i = 0; i < num_threads; i++) {
        if (times[i] > max) max = times[i];
    }
    return max;
}

__device__ __inline__ float
evaluateCuda(estimate_t* estimate, float x){
  return (estimate->b3)*x*x*x + (estimate->b2)*x*x + (estimate->b1)*x + estimate->b0;
}

__device__ __inline__ float
getdB0Cuda(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluateCuda(estimate, x);
  return -2.0 * (y-prediction) / static_cast<float>(N);
}

__device__ __inline__ float
getdB1Cuda(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluateCuda(estimate, x);
  return -2.0 * (y-prediction)*x / static_cast<float>(N);
}

__device__ __inline__ float
getdB2Cuda(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluateCuda(estimate, x);
  return -2.0 * (y-prediction)*x*x / static_cast<float>(N);
}

__device__ __inline__ float
getdB3Cuda(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluateCuda(estimate, x);
  return -2.0 * (y-prediction)*x*x*x / static_cast<float>(N);
}

__global__ void
setup_kernel(curandState *states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(418, index, 0, &states[index]);
}

__global__ void
sgd_cuda(estimate_t* estimates, float* x, float* y, curandState* states,
            float* times, int N, int totalThreads)
{
    clock_t start = clock();
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < NUM_ITER_STOCH; i++) {
        curandState localState = states[index];
        int pi = curand(&localState) % N;
        states[index] = localState;
        estimates[index].b0 -= STEP_SIZE_STOCH * getdB0Cuda(x[pi], y[pi], estimates+index, N);
        estimates[index].b1 -= STEP_SIZE_STOCH * getdB1Cuda(x[pi], y[pi], estimates+index, N);
        estimates[index].b2 -= STEP_SIZE_STOCH * getdB2Cuda(x[pi], y[pi], estimates+index, N);
        estimates[index].b3 -= STEP_SIZE_STOCH * getdB3Cuda(x[pi], y[pi], estimates+index, N);
    }
    clock_t end = clock();
    times[index] = static_cast<float>(end - start) / CLOCK_RATE;
}

// Running SGD with all threads each sampling one point and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdPerThread(int N, float* x, float* y, int blocks, int threadsPerBlock)
{
  float* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  int totalThreads = blocks * threadsPerBlock;

  cudaMalloc((void **)&device_times, totalThreads * sizeof(float));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * totalThreads);

  float* times = (float*)calloc(totalThreads, sizeof(float));
  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  for(int i = 0; i < totalThreads; i++) {
      initialize_estimate(estimates+i);
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_times, times, totalThreads * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, totalThreads * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);
  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  // Launch CUDA Kernel
  sgd_cuda<<<blocks, threadsPerBlock>>>(
      device_estimates, device_X, device_Y, states, device_times, N, totalThreads
  );
  cudaThreadSynchronize();

  cudaMemcpy(times, device_times, totalThreads * sizeof(float),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(estimates, device_estimates, totalThreads * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);

  average_estimates(estimates, ret, totalThreads);

  float max_time = 0.0;
  for(int j = 0; j < totalThreads; j++) {
      if (times[j] > max_time) max_time = times[j];
  }
  printf("Total execution time: %.3f seconds\n", max_time);

  return ret;
}

//-----------------------------------------------------------------------------

// Assumes the number of indexes is equal to N
__global__ void
sgd_by_block(estimate_t* estimates, float* x, float* y, curandState* states,
                float* times, int N, int blocks, int samplesPerThread)
{
  clock_t start = clock();
  __shared__ estimate_t thread_dbs[THREADS_PER_BLOCK];
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  times[blockIdx.x] = 0.0;
  clock_t end = clock();
  times[blockIdx.x] += static_cast<float>(end - start) / CLOCK_RATE;
  start = clock();

  for(int i = 0; i < NUM_ITER_STOCH_BLOCK; i++) {
      thread_dbs[threadIdx.x].b0 = 0.0;
      thread_dbs[threadIdx.x].b1 = 0.0;
      thread_dbs[threadIdx.x].b2 = 0.0;
      thread_dbs[threadIdx.x].b3 = 0.0;

      curandState localState = states[index];
      for(int j = 0; j < samplesPerThread; j++) {
          int pi = curand(&localState) % N;
          float db0 = getdB0Cuda(x[pi], y[pi], estimates+blockIdx.x, N);
          float db1 = getdB1Cuda(x[pi], y[pi], estimates+blockIdx.x, N);
          float db2 = getdB2Cuda(x[pi], y[pi], estimates+blockIdx.x, N);
          float db3 = getdB3Cuda(x[pi], y[pi], estimates+blockIdx.x, N);

          thread_dbs[threadIdx.x].b0 += db0 / static_cast<float>(samplesPerThread);
          thread_dbs[threadIdx.x].b1 += db1 / static_cast<float>(samplesPerThread);
          thread_dbs[threadIdx.x].b2 += db2 / static_cast<float>(samplesPerThread);
          thread_dbs[threadIdx.x].b3 += db3 / static_cast<float>(samplesPerThread);
      }
      states[index] = localState;

      __syncthreads();

      if(threadIdx.x == 0) {
          float db0 = 0.0;
          float db1 = 0.0;
          float db2 = 0.0;
          float db3 = 0.0;
          for(int k = 0; k < THREADS_PER_BLOCK; k++){
              db0 += thread_dbs[k].b0 / static_cast<float>(THREADS_PER_BLOCK);
              db1 += thread_dbs[k].b1 / static_cast<float>(THREADS_PER_BLOCK);
              db2 += thread_dbs[k].b2 / static_cast<float>(THREADS_PER_BLOCK);
              db3 += thread_dbs[k].b3 / static_cast<float>(THREADS_PER_BLOCK);
          }

          estimates[blockIdx.x].b0 -= STEP_SIZE_STOCH * db0;
          estimates[blockIdx.x].b1 -= STEP_SIZE_STOCH * db1;
          estimates[blockIdx.x].b2 -= STEP_SIZE_STOCH * db2;
          estimates[blockIdx.x].b3 -= STEP_SIZE_STOCH * db3;
          end = clock();
          times[blockIdx.x] += static_cast<float>(end - start) / CLOCK_RATE;
          start = clock();
      }
      __syncthreads();
  }
  if (threadIdx.x == 0) {
      clock_t end = clock();
      times[blockIdx.x] += static_cast<float>(end - start) / CLOCK_RATE;
  }
}

// Running SGD on each block with all threads sampling k points and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdCudaByBlock(int N, float* x, float* y, int samplesPerThread,
                           int blocks)
{
  float* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  int totalThreads = blocks * THREADS_PER_BLOCK;

  cudaMalloc((void **)&device_times, blocks * sizeof(float));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * blocks);

  float* times = (float*)calloc(totalThreads, sizeof(float));
  estimate_t* estimates = (estimate_t*)malloc(blocks * sizeof(estimate_t));
  for (int i = 0; i < blocks; i++) {
      initialize_estimate(estimates+i);
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_times, times, blocks * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, blocks * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, THREADS_PER_BLOCK>>>(states);

  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  sgd_by_block<<<blocks, THREADS_PER_BLOCK>>>(
      device_estimates, device_X, device_Y, states, device_times, N, blocks, samplesPerThread
  );
  cudaThreadSynchronize();

  cudaMemcpy(times, device_times, blocks * sizeof(float),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(estimates, device_estimates, blocks * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);

  average_estimates(estimates, ret, blocks);

  float max_time = 0.0;
  for(int j = 0; j < blocks; j++) {
      if (times[j] > max_time) max_time = times[j];
  }
  printf("Total execution time: %.3f seconds\n", max_time);

  return ret;
}

//-----------------------------------------------------------------------------

__device__ __inline__ void
shuffle(float* x, float* y, int size, curandState* randState){
  for(int i = 0; i < size; i++){
    int j = curand(randState) % size;

    float tempx = x[i];
    x[i] = x[j];
    x[j] = tempx;

    float tempy = y[i];
    y[i] = y[j];
    y[j] = tempy;
  }
}

// Assumes the number of indexes is equal to N
__global__ void
sgd_with_partition(estimate_t* estimates, float* x, float* y,
                    curandState* states, float* times, int N, int totalThreads)
{
  clock_t start = clock();
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  curandState localState = states[index];

  int bucketSize = N / totalThreads;
  int remainder = N % totalThreads;
  int lo, hi;

  if(index < remainder) {
    lo = index * bucketSize + index;
    hi = lo + bucketSize + 1;
  } else {
    lo = index * bucketSize + remainder;
    hi = lo + bucketSize;
  }

  int subsetSize = hi - lo;
  //Thread gets its subset of data based on threadIdx.x
  float* subsetX = (float*)malloc(sizeof(float) * subsetSize);
  float* subsetY = (float*)malloc(sizeof(float) * subsetSize);

  //copy data
  for(int i = lo; i < hi; i++)
  {
    subsetX[i-lo] = x[i];
    subsetY[i-lo] = y[i];
  }

  for (int i = 0; i < NUM_ITER_STOCH_PARTITION; i++) {
      shuffle(subsetX, subsetY, subsetSize, &localState);

      for(int j = 0; j < subsetSize; j++)
      {
          float db0 = getdB0Cuda(subsetX[j], subsetY[j], estimates+index, N);
          float db1 = getdB1Cuda(subsetX[j], subsetY[j], estimates+index, N);
          float db2 = getdB2Cuda(subsetX[j], subsetY[j], estimates+index, N);
          float db3 = getdB3Cuda(subsetX[j], subsetY[j], estimates+index, N);

          estimates[index].b0 -= (STEP_SIZE_STOCH * db0);
          estimates[index].b1 -= (STEP_SIZE_STOCH * db1);
          estimates[index].b2 -= (STEP_SIZE_STOCH * db2);
          estimates[index].b3 -= (STEP_SIZE_STOCH * db3);
      }
      states[index] = localState;
  }
  clock_t end = clock();
  times[index] = static_cast<float>(end - start) / CLOCK_RATE;
}

estimate_t* sgdCudaWithPartition(int N, float* x, float* y, int blocks,
                                 int threadsPerBlock)
{
  float* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  int totalThreads = blocks * threadsPerBlock;

  cudaMalloc((void **)&device_times, totalThreads * sizeof(float));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * totalThreads);

  float* times = (float*)calloc(totalThreads, sizeof(float));
  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  for(int i = 0; i < totalThreads; i++) {
      initialize_estimate(estimates+i);
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_times, times, totalThreads * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, totalThreads * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);
  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  sgd_with_partition<<<blocks, threadsPerBlock>>>(
      device_estimates, device_X, device_Y, states, device_times, N, totalThreads
  );
  cudaThreadSynchronize();

  cudaMemcpy(times, device_times, totalThreads * sizeof(float),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(estimates, device_estimates, totalThreads * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);

   average_estimates(estimates, ret, totalThreads);

   float max_time = 0.0;
   for(int j = 0; j < totalThreads; j++) {
       if (times[j] > max_time) max_time = times[j];
   }
   printf("Total execution time: %.3f seconds\n", max_time);

   return ret;
}
