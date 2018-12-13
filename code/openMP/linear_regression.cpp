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
    num_t* partial_db0 = (num_t*)malloc(sizeof(num_t) * num_threads);
		num_t* partial_db1 = (num_t*)malloc(sizeof(num_t) * num_threads);
    num_t* partial_db2 = (num_t*)malloc(sizeof(num_t) * num_threads);
    num_t* partial_db3 = (num_t*)malloc(sizeof(num_t) * num_threads);
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
    estimate -> b0 = INIT_B0;
	  estimate -> b1 = INIT_B1;
    estimate -> b2 = INIT_B2;
  	estimate -> b3 = INIT_B3;

	  for(int i = 0; i < NUM_ITER_BATCH; i++)
		{
				for(int k = 0; k < num_threads; k++) {
          partial_db0[k].num = 0.0;
					partial_db1[k].num = 0.0;
          partial_db2[k].num = 0.0;
          partial_db3[k].num = 0.0;
				}

				int j, tid;

		    #pragma omp parallel for default(shared) private(j, tid) schedule(static)
			    for(j = 0; j < N; j++)
			    {
				  	tid = omp_get_thread_num();
					  // TODO: change the partial sum arrays so they have padding
					  // 		right now, only a float so there is false sharing and so it is slower
            partial_db0[tid].num += getdB0(x[j], y[j], estimate, N);
					  partial_db1[tid].num += getdB1(x[j], y[j], estimate, N);
            partial_db2[tid].num += getdB2(x[j], y[j], estimate, N);
          	partial_db3[tid].num += getdB3(x[j], y[j], estimate, N);
			  	}

        float db0 = 0.0;
        float db1 = 0.0;
        float db2 = 0.0;
        float db3 = 0.0;
				for (int k = 0; k < num_threads; k++)
				{
          db0 += partial_db0[k].num;
					db1 += partial_db1[k].num;
          db2 += partial_db2[k].num;
          db3 += partial_db3[k].num;
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
      for(int i = 0; i < N; i++){
        sgd_step(N, x, y, estimate, i);
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
