#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <climits>

#include "gd.h"
#include "linear_regression.h"
#include "sgd_designs.h"

/*
			This design is one sgd estimate, but uses multiple threads to sample
      more points per update
 */
 estimate_t* sgd_with_k_samples(int N, float* x, float* y, int samplesPerThread,
                                int num_threads, double* time)
{
  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  estimate_t* estimate = (estimate_t*)(malloc(sizeof(estimate_t)));
  estimate -> b0 = INIT_B0;
  estimate -> b1 = INIT_B1;
  estimate -> b2 = INIT_B2;
  estimate -> b3 = INIT_B3;

  omp_set_num_threads(num_threads);
  int tid;

  unsigned int* seeds = (unsigned int*)malloc(sizeof(unsigned int) * num_threads);
  for(int i = 0; i < num_threads; i++){
    seeds[i] = (unsigned int)i;
  }

  float* local_db0 = (float*)malloc(sizeof(float) * num_threads);
  float* local_db1 = (float*)malloc(sizeof(float) * num_threads);
  float* local_db2 = (float*)malloc(sizeof(float) * num_threads);
  float* local_db3 = (float*)malloc(sizeof(float) * num_threads);

  for(int num_steps = 0; num_steps < NUM_ITER_STOCH; num_steps++){
    #pragma omp parallel default(shared) private(tid)
    {
      tid = omp_get_thread_num();
      local_db0[tid] = 0.0;
      local_db1[tid] = 0.0;
      local_db2[tid] = 0.0;
      local_db3[tid] = 0.0;

      for(int j = 0; j < samplesPerThread; j++){
        //sample random point
        int pi = rand_r(seeds+tid) % N;

        //compute gradient
        local_db0[tid] += getdB0(x[pi], y[pi], estimate, N) / static_cast<float>(samplesPerThread);
        local_db1[tid] += getdB1(x[pi], y[pi], estimate, N) / static_cast<float>(samplesPerThread);
        local_db2[tid] += getdB2(x[pi], y[pi], estimate, N) / static_cast<float>(samplesPerThread);
        local_db3[tid] += getdB3(x[pi], y[pi], estimate, N) / static_cast<float>(samplesPerThread);
      }
    }
  //
    //accumulate local_dbs
    float db0 = 0.0;
    float db1 = 0.0;
    float db2 = 0.0;
    float db3 = 0.0;

    for(int t = 0; t < num_threads; t++){
      db0 += local_db0[t] / static_cast<float>(num_threads);
      db1 += local_db1[t] / static_cast<float>(num_threads);
      db2 += local_db2[t] / static_cast<float>(num_threads);
      db3 += local_db3[t] / static_cast<float>(num_threads);
    }

    //update
    estimate -> b0 -= STEP_SIZE_STOCH * db0;
    estimate -> b1 -= STEP_SIZE_STOCH * db1;
    estimate -> b2 -= STEP_SIZE_STOCH * db2;
    estimate -> b3 -= STEP_SIZE_STOCH * db3;

    if(num_steps % 5000 == 0) {
          float MSE = calculate_error(N, x, y, estimate);
          printf("%.3f\n", MSE);
    }
  }

  return estimate;
}

//---------------------------------------------------------------------------

void averageEstimates(estimate_t* estimates, estimate_t* ret, int num_threads)
{
  ret -> b0 = 0.0;
  ret -> b1 = 0.0;
  ret -> b2 = 0.0;
  ret -> b3 = 0.0;
  for(int j = 0; j < num_threads; j++) {
    ret -> b0 += estimates[j].b0 / static_cast<float>(num_threads);
    ret -> b1 += estimates[j].b1 / static_cast<float>(num_threads);
    ret -> b2 += estimates[j].b2 / static_cast<float>(num_threads);
    ret -> b3 += estimates[j].b3 / static_cast<float>(num_threads);
  }
}
/*
			This design simply runs sgd on each thread and
			averages the results of b0 and b1 to compute the final answer
 */
estimate_t* sgd_per_thread(int N, float* x, float* y, int num_threads, double* time)
{

  using namespace std::chrono;
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> dsec;

  estimate_t* estimates = (estimate_t*)malloc(sizeof(estimate_t) * num_threads);

	for(int t = 0; t < num_threads; t++){
    estimates[t].b0 = INIT_B0;
		estimates[t].b1 = INIT_B1;
    estimates[t].b2 = INIT_B2;
    estimates[t].b3 = INIT_B3;
	}

	estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  omp_set_num_threads(num_threads);
	int tid;

  double* times = (double*)calloc(sizeof(double), num_threads);

  unsigned int* seeds = (unsigned int*)malloc(sizeof(unsigned int) * num_threads);
  for(int i = 0; i < num_threads; i++){
    seeds[i] = (unsigned int)i;
  }

  float MSE = calculate_error(N, x, y, estimates);
  printf("%.3f\n", MSE);

  // Run sgd in parallel to average the results
  #pragma omp parallel default(shared) private(tid)
	{
		tid = omp_get_thread_num();

    //shuffle
    float* shuffleX = (float*)(malloc(sizeof(float) * N));
    float* shuffleY = (float*)(malloc(sizeof(float) * N));
    for(int i = 0; i < N; i++){
      shuffleX[i] = x[i];
      shuffleY[i] = y[i];
    }

    shuffle(shuffleX, shuffleY, N, seeds+tid);

		for (int num_steps = 1; num_steps <= NUM_EPOCHS; num_steps++) {
      auto start = Clock::now();
      //1 epoch
      for(int i = 0; i < N; i++){
        sgd_step(N, shuffleX, shuffleY, estimates + tid, i);
      }
      auto end = Clock::now();
      times[tid] += duration_cast<dsec>(end - start).count();

      #pragma omp barrier

      if(tid == 0 && num_steps <= 20) {
            averageEstimates(estimates, ret, num_threads);
            float MSE = calculate_error(N, shuffleX, shuffleY, ret);
            printf("%.3f\n", MSE);
      }
    }
  }

  averageEstimates(estimates, ret, num_threads);
  double max_time;
  for(int j = 0; j < num_threads; j++) {
    if(times[j] > max_time){
      max_time = times[j];
    }
  }

  *time = max_time;

  return ret;
}
