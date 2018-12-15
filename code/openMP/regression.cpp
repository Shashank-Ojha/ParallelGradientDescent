#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <climits>

#include "gd.h"
#include "regression.h"

float evaluate(estimate_t* estimate, float x){
  return (estimate->b3)*x*x*x + (estimate->b2)*x*x + (estimate->b1)*x + estimate->b0;
}

float getdB3(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x*x*x / static_cast<float>(N);
}

float getdB2(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x*x / static_cast<float>(N);
}

float getdB1(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x / static_cast<float>(N);
}

float getdB0(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction) / static_cast<float>(N);
}

float calculate_error(int N, float* x, float* y, estimate_t* estimate) {
	float res = 0.0;
	for (int i = 0; i < N; i++) {
    float y_hat = evaluate(estimate, x[i]);
		res += ((y[i] - y_hat) * (y[i] - y_hat)) / static_cast<float>(N);
	}

	return res;
}

estimate_t* bgd(int N, float* x, float* y, int num_threads)
{
	  omp_set_num_threads(num_threads);
    float* partial_db0 = (float*)malloc(sizeof(float) * num_threads);
		float* partial_db1 = (float*)malloc(sizeof(float) * num_threads);
    float* partial_db2 = (float*)malloc(sizeof(float) * num_threads);
    float* partial_db3 = (float*)malloc(sizeof(float) * num_threads);
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
    estimate -> b0 = INIT_B0;
	  estimate -> b1 = INIT_B1;
    estimate -> b2 = INIT_B2;
  	estimate -> b3 = INIT_B3;

	  for(int i = 0; i < NUM_ITER_BATCH; i++)
		{
				int j, tid;
		    #pragma omp parallel for default(shared) private(j, tid) schedule(static)
			    for(j = 0; j < N; j++)
			    {
				  	tid = omp_get_thread_num();
            partial_db0[tid] = 0.0;
            partial_db1[tid] = 0.0;
            partial_db2[tid] = 0.0;
            partial_db3[tid] = 0.0;

            partial_db0[tid] += getdB0(x[j], y[j], estimate, N) / static_cast<float>(N);
					  partial_db1[tid] += getdB1(x[j], y[j], estimate, N) / static_cast<float>(N);
            partial_db2[tid] += getdB2(x[j], y[j], estimate, N) / static_cast<float>(N);
          	partial_db3[tid] += getdB3(x[j], y[j], estimate, N) / static_cast<float>(N);
			  	}

        float db0 = 0.0;
        float db1 = 0.0;
        float db2 = 0.0;
        float db3 = 0.0;
				for (int k = 0; k < num_threads; k++)
				{
          db0 += partial_db0[k];
					db1 += partial_db1[k];
          db2 += partial_db2[k];
          db3 += partial_db3[k];
				}

        estimate -> b0 -= STEP_SIZE_BATCH * db0;
		    estimate -> b1 -= STEP_SIZE_BATCH * db1;
        estimate -> b2 -= STEP_SIZE_BATCH * db2;
    		estimate -> b3 -= STEP_SIZE_BATCH * db3;
  	}
  	return estimate;
}

void sgd_step(int N, float* x, float* y, estimate_t* estimate, int j)
{
    j = j % N;
    estimate -> b0 -= STEP_SIZE_STOCH * getdB0(x[j], y[j], estimate, N);
    estimate -> b1 -= STEP_SIZE_STOCH * getdB1(x[j], y[j], estimate, N);
    estimate -> b2 -= STEP_SIZE_STOCH * getdB2(x[j], y[j], estimate, N);
    estimate -> b3 -= STEP_SIZE_STOCH * getdB3(x[j], y[j], estimate, N);
}

estimate_t* sgd(int N, float* x, float* y)
{
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b0 = INIT_B0;
    estimate -> b1 = INIT_B1;
    estimate -> b2 = INIT_B2;
    estimate -> b3 = INIT_B3;

	  for(int i = 0; i < NUM_ITER_STOCH; i++){
        int pi = rand() % N;
        sgd_step(N, x, y, estimate, pi);
  	}
  	return estimate;
}

estimate_t* sgd_epochs(int N, float* x, float* y)
{
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b0 = INIT_B0;
    estimate -> b1 = INIT_B1;
    estimate -> b2 = INIT_B2;
    estimate -> b3 = INIT_B3;

	  for(int i = 0; i < NUM_EPOCHS; i++){
      for(int j = 0; j < N; j++){
        sgd_step(N, x, y, estimate, j);
      }
  	}
  	return estimate;
}


void shuffle(float* x, float* y, int N, unsigned int* tid_seed){
  for(int i = 0; i < N; i++){
    int j = rand_r(tid_seed) % N;

    float tempx = x[i];
    x[i] = x[j];
    x[j] = tempx;

    float tempy = y[i];
    y[i] = y[j];
    y[j] = tempy;
  }
}
