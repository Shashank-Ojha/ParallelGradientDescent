#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "regression.h"

#define THREADS_PER_BLOCK 8

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

// void shuffle(float* x, float* y, int N, unsigned int* tid_seed){
//   for(int i = 0; i < N; i++){
//     int j = rand_r(tid_seed) % N;
//
//     float tempx = x[i];
//     x[i] = x[j];
//     x[j] = tempx;
//
//     float tempy = y[i];
//     y[i] = y[j];
//     y[j] = tempy;
//   }
// }

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
sgd_cuda(estimate_t* estimates, estimate_t* benchmarks, long* benchmark_times,
            float* x, float* y, curandState* states, long* times, int N, int totalThreads)
{
    clock_t start, end;
    start = clock();
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < NUM_ITER_STOCH; i++) {
        curandState localState = states[index];
        int pi = curand(&localState) % N;
        states[index] = localState;
        estimates[index].b0 -= STEP_SIZE_STOCH * getdB0Cuda(x[pi], y[pi], estimates+index, N);
        estimates[index].b1 -= STEP_SIZE_STOCH * getdB1Cuda(x[pi], y[pi], estimates+index, N);
        estimates[index].b2 -= STEP_SIZE_STOCH * getdB2Cuda(x[pi], y[pi], estimates+index, N);
        estimates[index].b3 -= STEP_SIZE_STOCH * getdB3Cuda(x[pi], y[pi], estimates+index, N);

        end = clock();
        times[index] += (long)(end - start);
        int interval = NUM_ITER_STOCH / NUM_CHECKS;
        if (i % interval == 0) {
            benchmarks[(i/interval)*totalThreads + index] = estimates[index];
            benchmark_times[(i/interval)*totalThreads + index] = times[index];
        }
        start = clock();
    }
}

// Running SGD with all threads each sampling one point and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdPerThread(int N, float* x, float* y, int blocks, int threadsPerBlock)
{
  long* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;
  estimate_t* device_benchmarks;
  long* device_benchmark_times;

  int totalThreads = blocks * threadsPerBlock;
  int numBenchmarks = NUM_CHECKS * totalThreads;

  cudaMalloc((void **)&device_times, totalThreads * sizeof(long));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * totalThreads);
  cudaMalloc((void **)&device_benchmarks, sizeof(estimate_t) * numBenchmarks);
  cudaMalloc((void **)&device_benchmark_times, sizeof(long) * numBenchmarks);

  long* times = (long*)calloc(totalThreads, sizeof(long));
  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  estimate_t* benchmarks = (estimate_t*)calloc(numBenchmarks, sizeof(estimate_t));
  long* benchmark_times = (long*)calloc(numBenchmarks, sizeof(long));
  for(int i = 0; i < totalThreads; i++) {
      initialize_estimate(estimates+i);
  }
  for(int j = 0; j < numBenchmarks; j++) {
      initialize_estimate(benchmarks+j);
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_times, times, totalThreads * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, totalThreads * sizeof(estimate_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_benchmarks, benchmarks, numBenchmarks * sizeof(estimate_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_benchmark_times, benchmark_times, numBenchmarks * sizeof(long),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);
  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  // Launch CUDA Kernel
  sgd_cuda<<<blocks, threadsPerBlock>>>(
      device_estimates, device_benchmarks, device_benchmark_times,
      device_X, device_Y, states, device_times, N, totalThreads
  );
  cudaThreadSynchronize();

  cudaMemcpy(times, device_times, totalThreads * sizeof(long),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(estimates, device_estimates, totalThreads * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(benchmarks, device_benchmarks, numBenchmarks * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(benchmark_times, device_benchmark_times, numBenchmarks * sizeof(long),
               cudaMemcpyDeviceToHost);

  // Print the MSE at various iterations
  printf("MSE\n\n");
  for(int k = 0; k < NUM_CHECKS; k++) {
      initialize_estimate(ret);
      average_estimates(benchmarks+(totalThreads*k), ret, totalThreads);
      float MSE = calculate_error(N, x, y, ret);
      // printf("Steps: %d\tMSE: %.3f\n", k*(NUM_ITER_STOCH / NUM_CHECKS), MSE);
      printf("%.3f\n", MSE);
  }

  // Print the number of clock cycles at the checkpoints
  printf("\n\nClock cycles\n\n");
  for(int l = 0; l < NUM_CHECKS; l++) {
      long time = get_max_time(benchmark_times+(totalThreads*l), totalThreads);
      // printf("Steps: %d\tClock Cycles: %ld\n", l*(NUM_ITER_STOCH / NUM_CHECKS), time);
      printf("%ld\n", time);
  }

  long max_time = 0;
  for(int l = 0; l < totalThreads; l++) {
      if (times[l] > max_time) max_time = times[l];
  }
  printf("Num of clock cycles: %ld\n", max_time);

  return ret;
}

//-----------------------------------------------------------------------------

// Assumes the number of indexes is equal to N
__global__ void
sgd_by_block(estimate_t* estimates, estimate_t* benchmarks, long* benchmark_times,
            float* x, float* y, curandState* states, long* times, int N, int blocks,
            int samplesPerThread)
{
  clock_t start, end;
  start = clock();

  __shared__ estimate_t thread_dbs[THREADS_PER_BLOCK];
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  for(int i = 0; i < NUM_ITER_STOCH; i++) {
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

          int interval = NUM_ITER_STOCH / NUM_CHECKS;
          if (i % interval == 0) {
              end = clock();
              benchmarks[(i/interval)*blocks + blockIdx.x] = estimates[blockIdx.x];
              benchmark_times[(i/interval)*blocks + blockIdx.x] = (long)(end - start);
          }
      }
      __syncthreads();
  }

  end = clock();
  times[blockIdx.x] = (long)(end - start);
}

// Running SGD on each block with all threads sampling k points and averaging result
// after each SGD step. Checking convergence after each step
estimate_t* sgdCudaByBlock(int N, float* x, float* y, int samplesPerThread,
                           int blocks)
{
  long* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;
  estimate_t* device_benchmarks;
  long* device_benchmark_times;

  int totalThreads = blocks * THREADS_PER_BLOCK;
  int numBenchmarks = NUM_CHECKS * blocks;

  cudaMalloc((void **)&device_times, totalThreads * sizeof(long));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * blocks);
  cudaMalloc((void **)&device_benchmarks, sizeof(estimate_t) * numBenchmarks);
  cudaMalloc((void **)&device_benchmark_times, sizeof(long) * numBenchmarks);

  long* times = (long*)calloc(totalThreads, sizeof(long));
  estimate_t* estimates = (estimate_t*)malloc(blocks * sizeof(estimate_t));
  estimate_t* benchmarks = (estimate_t*)malloc(numBenchmarks * sizeof(estimate_t));
  long* benchmark_times = (long*)calloc(numBenchmarks, sizeof(long));
  for (int i = 0; i < blocks; i++) {
      initialize_estimate(estimates+i);
  }
  for(int j = 0; j < numBenchmarks; j++) {
      initialize_estimate(benchmarks+j);
  }

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_times, times, totalThreads * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, blocks * sizeof(estimate_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_benchmarks, benchmarks, numBenchmarks * sizeof(estimate_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_benchmark_times, benchmark_times, numBenchmarks * sizeof(long),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, THREADS_PER_BLOCK>>>(states);

  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  sgd_by_block<<<blocks, THREADS_PER_BLOCK>>>(
      device_estimates, device_benchmarks, device_benchmark_times,
      device_X, device_Y, states, device_times, N, blocks, samplesPerThread
  );
  cudaThreadSynchronize();

  cudaMemcpy(times, device_times, totalThreads * sizeof(long),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(estimates, device_estimates, blocks * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(benchmarks, device_benchmarks, numBenchmarks * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(benchmark_times, device_benchmark_times, numBenchmarks * sizeof(long),
               cudaMemcpyDeviceToHost);


  // Print the MSE at various iterations
  printf("MSE\n\n");
  for(int k = 0; k < NUM_CHECKS; k++) {
      average_estimates(benchmarks+(blocks*k), ret, blocks);
      float MSE = calculate_error(N, x, y, ret);
      // printf("Steps: %d\tMSE: %.3f\n", k*(NUM_ITER_STOCH / NUM_CHECKS), MSE);
      printf("%.3f\n", MSE);
  }

  // Print the number of clock cycles at the checkpoints
  printf("\n\nClock cycles\n\n");
  for(int l = 0; l < NUM_CHECKS; l++) {
      long time = get_max_time(benchmark_times+(blocks*l), blocks);
      // printf("Steps: %d\tClock Cycles: %ld\n", l*(NUM_ITER_STOCH / NUM_CHECKS), time);
      printf("%ld\n", time);
  }

  long max_time = 0;
  for(int l = 0; l < blocks; l++) {
      if (times[l] > max_time) max_time = times[l];
  }

  ret->b0 = 0.0;
  ret->b1 = 0.0;
  ret->b2 = 0.0;
  ret->b3 = 0.0;
  printf("Num of clock cycles: %ld\n", max_time);
  for(int i = 0; i < blocks; i++) {
    ret -> b0 += (estimates+i) -> b0 / static_cast<float>(blocks);
    ret -> b1 += (estimates+i) -> b1 / static_cast<float>(blocks);
    ret -> b2 += (estimates+i) -> b2 / static_cast<float>(blocks);
    ret -> b3 += (estimates+i) -> b3 / static_cast<float>(blocks);
    printf("y = (%.5f) x^3 + (%.5f) x^2 + (%.5f) x + (%.5f)\n",
           (estimates+i) -> b3, (estimates+i) -> b2, (estimates+i) -> b1, (estimates+i) -> b0);
  }
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
sgd_with_partition(estimate_t* estimates, estimate_t* benchmarks, long* benchmark_times,
            float* x, float* y, curandState* states, long* times, int N, int totalThreads)
{
  clock_t start = clock();
  clock_t end;
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

  for (int i = 0; i < NUM_ITER_STOCH; i++) {
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
      int interval = NUM_ITER_STOCH / NUM_CHECKS;
      if (i % interval == 0) {
          end = clock();
          benchmarks[(i/interval)*totalThreads + index] = estimates[index];
          benchmark_times[(i/interval)*totalThreads + index] = (long)(end - start);
      }
  }
  clock_t stop = clock();
  times[index] = (long)(stop - start);
}

estimate_t* sgdCudaWithPartition(int N, float* x, float* y, int blocks,
                                 int threadsPerBlock)
{
  long* device_times;
  curandState* states;
  float* device_X;
  float* device_Y;
  estimate_t* device_estimates;
  estimate_t* device_benchmarks;
  long* device_benchmark_times;

  int totalThreads = blocks * threadsPerBlock;
  int numBenchmarks = NUM_CHECKS * totalThreads;

  cudaMalloc((void **)&device_times, totalThreads * sizeof(long));
  cudaMalloc((void **)&states, totalThreads * sizeof(curandState));
  cudaMalloc((void **)&device_X, sizeof(float) * N);
  cudaMalloc((void **)&device_Y, sizeof(float) * N);
  cudaMalloc((void **)&device_estimates, sizeof(estimate_t) * totalThreads);
  cudaMalloc((void **)&device_benchmarks, sizeof(estimate_t) * numBenchmarks);
  cudaMalloc((void **)&device_benchmark_times, sizeof(long) * numBenchmarks);

  long* times = (long*)calloc(totalThreads, sizeof(long));
  estimate_t* estimates = (estimate_t*)calloc(totalThreads, sizeof(estimate_t));
  estimate_t* benchmarks = (estimate_t*)calloc(numBenchmarks, sizeof(estimate_t));
  long* benchmark_times = (long*)calloc(numBenchmarks, sizeof(long));
  for(int i = 0; i < totalThreads; i++) {
      initialize_estimate(estimates+i);
  }
  for(int j = 0; j < numBenchmarks; j++) {
      initialize_estimate(benchmarks+j);
  }

  // Randomly shuffle the data once so the partitions are random
  // unsigned int seed = 15418;
  // shuffle(x, y, N, &seed);

  cudaMemcpy(device_X, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_Y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(device_times, times, totalThreads * sizeof(long), cudaMemcpyHostToDevice);
  cudaMemcpy(device_estimates, estimates, totalThreads * sizeof(estimate_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_benchmarks, benchmarks, numBenchmarks * sizeof(estimate_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_benchmark_times, benchmark_times, numBenchmarks * sizeof(long),
             cudaMemcpyHostToDevice);

  setup_kernel<<<blocks, threadsPerBlock>>>(states);
  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  sgd_with_partition<<<blocks, threadsPerBlock>>>(
      device_estimates, device_benchmarks, device_benchmark_times,
      device_X, device_Y, states, device_times, N, totalThreads
  );
  cudaThreadSynchronize();

  cudaMemcpy(times, device_times, totalThreads * sizeof(long),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(estimates, device_estimates, totalThreads * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(benchmarks, device_benchmarks, numBenchmarks * sizeof(estimate_t),
               cudaMemcpyDeviceToHost);
  cudaMemcpy(benchmark_times, device_benchmark_times, numBenchmarks * sizeof(long),
               cudaMemcpyDeviceToHost);

   // Print the MSE at various iterations
   printf("MSE\n\n");
   for(int k = 0; k < NUM_CHECKS; k++) {
       average_estimates(benchmarks+(totalThreads*k), ret, totalThreads);
       float MSE = calculate_error(N, x, y, ret);
       // printf("Steps: %d\tMSE: %.3f\n", k*(NUM_ITER_STOCH / NUM_CHECKS), MSE);
       printf("%.3f\n", MSE);
   }

   // Print the number of clock cycles at the checkpoints
   printf("\n\nClock cycles\n\n");
   for(int l = 0; l < NUM_CHECKS; l++) {
       long time = get_max_time(benchmark_times+(totalThreads*l), totalThreads);
       // printf("Steps: %d\tClock Cycles: %ld\n", l*(NUM_ITER_STOCH / NUM_CHECKS), time);
       printf("%ld\n", time);
   }

   long max_time = 0;
   for(int l = 0; l < totalThreads; l++) {
       if (times[l] > max_time) max_time = times[l];
   }
   printf("Num of clock cycles: %ld\n", max_time);

   return ret;
}
