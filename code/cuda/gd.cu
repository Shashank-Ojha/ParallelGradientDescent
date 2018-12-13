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


float evaluate(estimate_t* estimate, float x){
  return (estimate->b3)*x*x*x + (estimate->b2)*x*x + (estimate->b1)*x + estimate->b0;
}

float calculateMSE(estimate_t* est, float* X, float* Y, int N) {
    float est_MSE = 0.0;
    for (int i = 0; i < N; i++) {
        float error = Y[i] - evaluate(est, X[i]);
        est_MSE += error * error / static_cast<float>(N);
    }

    return est_MSE;
}

__device__ __inline__ float
evaluateCuda(estimate_t* estimate, float x){
  return (estimate->b3)*x*x*x + (estimate->b2)*x*x + (estimate->b1)*x + estimate->b0;
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

__device__ __inline__ float
getdB2Cuda(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluateCuda(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction)*x*x;
}

__device__ __inline__ float
getdB3Cuda(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluateCuda(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction)*x*x;
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
  float db2 = getdB0Cuda(device_X[pi], device_Y[pi], device_estimates+index, N);
  float db3 = getdB1Cuda(device_X[pi], device_Y[pi], device_estimates+index, N);

  device_estimates[index].b0 -= STEP_SIZE_STOCH * db0;
  device_estimates[index].b1 -= STEP_SIZE_STOCH * db1;
  device_estimates[index].b2 -= STEP_SIZE_STOCH * db2;
  device_estimates[index].b3 -= STEP_SIZE_STOCH * db3;

  clock_t end = clock();
  times[index] = (int)(end - start);
}

// Running SGD with all threads each sampling one point and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdCuda(int N, float* x, float* y, int blocks, int threadsPerBlock)
{
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
      estimates[i].b2 = INIT_B2;
      estimates[i].b3 = INIT_B3;
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, totalThreads * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);

  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  int times[totalThreads];
  int total_time = 0;
  for (int num_iters = 0; num_iters < NUM_ITER_STOCH; num_iters++) {
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
    ret -> b2 = 0.0;
    ret -> b3 = 0.0;
    for(int j = 0; j < totalThreads; j++) {
      ret -> b0 += estimates[j].b0 / static_cast<float>(totalThreads);
      ret -> b1 += estimates[j].b1 / static_cast<float>(totalThreads);
      ret -> b2 += estimates[j].b2 / static_cast<float>(totalThreads);
      ret -> b3 += estimates[j].b3 / static_cast<float>(totalThreads);
    }

    if (num_iters == 25 || num_iters == 100 || num_iters == 250 ||
        num_iters == 500 || num_iters == 1000 || num_iters == 1500 ||
        num_iters == 2000 || num_iters == 2500 || num_iters == 5000) {
        float MSE = calculateMSE(ret, x, y, N);
        printf("Steps: %d\tMSE: %.3f\n", num_iters, MSE);
    }

  }

  printf("Num of clock cycles: %d\n", total_time);

  return ret;
}

//-----------------------------------------------------------------------------

// Assumes the number of indexes is equal to N
__global__ void
sgdStepByBlock(int N, float* device_X, float* device_Y,
               estimate_t* device_estimates, int samplesPerThread, curandState* states, int* times) {
  clock_t start = clock();
  __shared__ estimate_t thread_estimate[THREADS_PER_BLOCK];

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  thread_estimate[threadIdx.x].b0 = INIT_B0;
  thread_estimate[threadIdx.x].b1 = INIT_B1;
  thread_estimate[threadIdx.x].b2 = INIT_B2;
  thread_estimate[threadIdx.x].b3 = INIT_B3;

  curandState localState = states[index];

  for(int i = 0; i < samplesPerThread; i++)
  {
      int pi = curand(&localState) % N;
      thread_estimate[threadIdx.x].b0 +=
        (getdB0Cuda(device_X[pi], device_Y[pi], device_estimates+blockIdx.x, N) /
        static_cast<float>(samplesPerThread));
      thread_estimate[threadIdx.x].b1 +=
        (getdB1Cuda(device_X[pi], device_Y[pi], device_estimates+blockIdx.x, N) /
        static_cast<float>(samplesPerThread));
      thread_estimate[threadIdx.x].b2 +=
        (getdB2Cuda(device_X[pi], device_Y[pi], device_estimates+blockIdx.x, N) /
        static_cast<float>(samplesPerThread));
      thread_estimate[threadIdx.x].b3 +=
        (getdB3Cuda(device_X[pi], device_Y[pi], device_estimates+blockIdx.x, N) /
        static_cast<float>(samplesPerThread));
  }
  states[index] = localState;

  __syncthreads();

  if(threadIdx.x == 0)
  {
      float db0 = 0.0;
      float db1 = 0.0;
      float db2 = 0.0;
      float db3 = 0.0;
      for(int i = 0; i < blockDim.x; i++){
          db0 += thread_estimate[i].b0 / static_cast<float>(blockDim.x);
          db1 += thread_estimate[i].b1 / static_cast<float>(blockDim.x);
          db2 += thread_estimate[i].b2 / static_cast<float>(blockDim.x);
          db3 += thread_estimate[i].b3 / static_cast<float>(blockDim.x);
      }

      device_estimates[blockIdx.x].b0 -= STEP_SIZE_STOCH * db0;
      device_estimates[blockIdx.x].b1 -= STEP_SIZE_STOCH * db1;
      device_estimates[blockIdx.x].b2 -= STEP_SIZE_STOCH * db2;
      device_estimates[blockIdx.x].b3 -= STEP_SIZE_STOCH * db3;
  }
  clock_t stop = clock();
  times[index] = (int)(stop - start);
}

// Running SGD on each block with all threads sampling k points and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdCudaByBlock(int N, float* x, float* y, int samplesPerThread,
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
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * blocks);

  estimate_t* estimates = (estimate_t*)calloc(blocks, sizeof(estimate_t));
  for (int i = 0; i < blocks; i++) {
      estimates[i].b0 = INIT_B0;
      estimates[i].b1 = INIT_B1;
      estimates[i].b2 = INIT_B2;
      estimates[i].b3 = INIT_B3;
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, blocks * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);

  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  int times[totalThreads];
  int total_time = 0;

  for (int num_iters = 0; num_iters < NUM_ITER_STOCH; num_iters++) {
    sgdStepByBlock<<<blocks, threadsPerBlock>>>(N, device_X, device_Y, device_estimates, samplesPerThread, states, device_times);
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
    ret -> b2 = 0.0;
    ret -> b3 = 0.0;
    for(int j = 0; j < blocks; j++) {
      ret -> b0 += estimates[j].b0 / static_cast<float>(blocks);
      ret -> b1 += estimates[j].b1 / static_cast<float>(blocks);
      ret -> b2 += estimates[j].b2 / static_cast<float>(blocks);
      ret -> b3 += estimates[j].b3 / static_cast<float>(blocks);
    }

    if (num_iters == 25 || num_iters == 100 || num_iters == 250 ||
        num_iters == 500 || num_iters == 1000 || num_iters == 1500 ||
        num_iters == 2000 || num_iters == 2500 || num_iters == 5000) {
        float MSE = calculateMSE(ret, x, y, N);
        printf("Steps: %d\tMSE: %.3f\n", num_iters, MSE);
    }
  }

  printf("Num of clock cycles: %d\n", total_time);

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
sgdStepWithPartition(int N, float* device_X, float* device_Y,
               estimate_t* device_estimates, curandState* states, long* times) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  curandState localState = states[index];

  int bucketSize = N / blockDim.x;
  int remainder = N % blockDim.x;
  int lo;
  int hi;

  //thread 0 -> lo = 0 and hi = 4
  if(threadIdx.x < remainder)
  {
    lo = threadIdx.x * bucketSize + threadIdx.x;
    hi = lo + bucketSize + 1;
  } else {
    lo = threadIdx.x * bucketSize + remainder;
    hi = lo + bucketSize;
  }

  int subsetSize = hi - lo;
  //Thread gets its subset of data based on threadIdx.x
  float* subsetX = (float*)malloc(sizeof(float) * subsetSize);
  float* subsetY = (float*)malloc(sizeof(float) * subsetSize);

  //copy data
  for(int i = lo; i < hi; i++)
  {
    subsetX[i-lo] = device_X[i];
    subsetY[i-lo] = device_Y[i];
  }

  clock_t start = clock();

  shuffle(subsetX, subsetY, subsetSize, &localState);


  for(int i = 0; i < subsetSize; i++)
  {
      float db0 = getdB0Cuda(subsetX[i], subsetY[i], device_estimates+index, N);
      float db1 = getdB1Cuda(subsetX[i], subsetY[i], device_estimates+index, N);
      float db2 = getdB2Cuda(subsetX[i], subsetY[i], device_estimates+index, N);
      float db3 = getdB3Cuda(subsetX[i], subsetY[i], device_estimates+index, N);

      device_estimates[index].b0 -= (STEP_SIZE_STOCH * db0);
      device_estimates[index].b1 -= (STEP_SIZE_STOCH * db1);
      device_estimates[index].b2 -= (STEP_SIZE_STOCH * db2);
      device_estimates[index].b3 -= (STEP_SIZE_STOCH * db3);
  }

  states[index] = localState;

  clock_t stop = clock();
  times[index] = (long)(stop - start);
}

estimate_t* sgdCudaWithPartition(int N, float* x, float* y, int blocks,
                                 int threadsPerBlock){

  long* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;

  int totalThreads = blocks * threadsPerBlock;

  cudaMalloc((void **)&device_times, totalThreads * sizeof(long));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * totalThreads);

  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  for (int i = 0; i < totalThreads; i++) {
      estimates[i].b0 = INIT_B0;
      estimates[i].b1 = INIT_B1;
      estimates[i].b2 = INIT_B2;
      estimates[i].b3 = INIT_B3;
  }

  //TODO: shuffle x and y

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, totalThreads * sizeof(estimate_t),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);

  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  long times[totalThreads];
  long total_time = 0;

  for (int num_iters = 0; num_iters < NUM_ITER_STOCH; num_iters++) {
    sgdStepWithPartition<<<blocks, threadsPerBlock>>>(N, device_X, device_Y, device_estimates, states, device_times);
    cudaThreadSynchronize();

    cudaMemcpy(times, device_times, totalThreads * sizeof(long),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(estimates, device_estimates, totalThreads * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);

    long max_time = 0;
    for (int i = 0; i < totalThreads; i++) {
        if (times[i] > max_time) max_time = times[i];
    }

    total_time += max_time;

    ret -> b0 = 0.0;
    ret -> b1 = 0.0;
    ret -> b2 = 0.0;
    ret -> b3 = 0.0;
    for(int j = 0; j < totalThreads; j++) {
      ret -> b0 += estimates[j].b0 / static_cast<float>(totalThreads);
      ret -> b1 += estimates[j].b1 / static_cast<float>(totalThreads);
      ret -> b2 += estimates[j].b2 / static_cast<float>(totalThreads);
      ret -> b3 += estimates[j].b3 / static_cast<float>(totalThreads);
    }

    if (num_iters == 25 || num_iters == 100 || num_iters == 250 ||
        num_iters == 500 || num_iters == 1000 || num_iters == 1500 ||
        num_iters == 2000 || num_iters == 2500 || num_iters == 5000) {
        float MSE = calculateMSE(ret, x, y, N);
        printf("Steps: %d\tMSE: %.3f\n", num_iters, MSE);
    }

  }

  printf("Num of clock cycles: %ld\n", total_time);

  return ret;
}
