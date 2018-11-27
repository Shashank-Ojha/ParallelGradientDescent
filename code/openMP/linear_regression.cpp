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
  return (estimate->b1)*x + (estimate->b0);
}

float getdB0(float x, float y, estimate_t* estimate){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction);
}

float getdB1(float x, float y, estimate_t* estimate){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x;
}

float calculate_error(int N, float* x, float* y, estimate_t estimate) {
	float res = 0;
	float b0 = estimate.b0;
	float b1 = estimate.b1;
	for (int i = 0; i < N; i++) {
		res += (y[i] - b0 - x[i] * b1) * (y[i] - b0 - x[i] * b1) / static_cast<float>(N);
	}

	return res;
}

estimate_t* bgd(int N, float* x, float* y, int num_threads)
{
	  omp_set_num_threads(num_threads);
		num_t* partial_db0 = (num_t*)malloc(sizeof(num_t) * num_threads);
		num_t* partial_db1 = (num_t*)malloc(sizeof(num_t) * num_threads);
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b0 = 0.0;
	  estimate -> b1 = 0.0;

	  for(int i = 0; i < NUM_ITER_BATCH; i++)
		{
				for(int k = 0; k < num_threads; k++) {
					partial_db0[k].num = 0;
					partial_db1[k].num = 0;
				}

		    float db0 = 0;
		    float db1 = 0;
				int j, tid;

		    #pragma omp parallel for default(shared) private(j, tid) schedule(static)
			    for(j = 0; j < N; j++)
			    {
			      float local_db0 = (1.0 / static_cast<float>(N)) * getdB0(x[j], y[j], estimate);
			      float local_db1 = (1.0 / static_cast<float>(N)) * getdB1(x[j], y[j], estimate);
				  	tid = omp_get_thread_num();
					  // TODO: change the partial sum arrays so they have padding
					  // 		right now, only a float so there is false sharing and so it is slower
					  partial_db0[tid].num += local_db0;
					  partial_db1[tid].num += local_db1;
			  	}

				for (int k = 0; k < num_threads; k++)
				{
					db0 += partial_db0[k].num;
					db1 += partial_db1[k].num;
				}

		    estimate -> b0 = (estimate -> b0) - (STEP_SIZE_BATCH * db0);
		    estimate -> b1 = (estimate -> b1) - (STEP_SIZE_BATCH * db1);
  	}
  	return estimate;
}

estimate_t* sgd(int N, float* x, float* y)
{
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b0 = 0.0;
	  estimate -> b1 = 0.0;

	  for(int i = 0; i < NUM_ITER_BATCH; i++)
		{
        int j = rand() % N;

		    float db0 = (1.0 / static_cast<float>(N)) * getdB0(x[j], y[j], estimate);
		    float db1 = (1.0 / static_cast<float>(N)) * getdB1(x[j], y[j], estimate);

		    estimate -> b0 = (estimate -> b0) - (STEP_SIZE_BATCH * db0);
		    estimate -> b1 = (estimate -> b1) - (STEP_SIZE_BATCH * db1);
  	}
  	return estimate;
}
