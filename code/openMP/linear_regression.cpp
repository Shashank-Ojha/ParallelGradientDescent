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
  return (estimate->b1)*x;
}

float getdB1(float x, float y, estimate_t* estimate){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x;
}

float calculate_error(int N, float* x, float* y, estimate_t estimate) {
	float res = 0;
	float b1 = estimate.b1;
	for (int i = 0; i < N; i++) {
		res += (y[i] - x[i] * b1) * (y[i] - x[i] * b1);
	}

	return res / static_cast<float>(N);
}

estimate_t* bgd(int N, float* x, float* y, int num_threads)
{
	  omp_set_num_threads(num_threads);
		num_t* partial_db1 = (num_t*)malloc(sizeof(num_t) * num_threads);
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b1 = 0.0;

	  for(int i = 0; i < NUM_ITER_BATCH; i++)
		{
				for(int k = 0; k < num_threads; k++) {
					partial_db1[k].num = 0;
				}

		    float db1 = 0;
				int j, tid;

		    #pragma omp parallel for default(shared) private(j, tid) schedule(static)
			    for(j = 0; j < N; j++)
			    {
			      float local_db1 = (1.0 / static_cast<float>(N)) * getdB1(x[j], y[j], estimate);
				  	tid = omp_get_thread_num();
					  // TODO: change the partial sum arrays so they have padding
					  // 		right now, only a float so there is false sharing and so it is slower
					  partial_db1[tid].num += local_db1;
			  	}

				for (int k = 0; k < num_threads; k++)
				{
					db1 += partial_db1[k].num;
				}

		    estimate -> b1 = (estimate -> b1) - (STEP_SIZE_BATCH * db1);
  	}
  	return estimate;
}

estimate_t* sgd(int N, float* x, float* y)
{
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b1 = 0.0;

	  for(int i = 0; i < NUM_ITER_BATCH; i++)
		{
        int j = rand() % N;

		    float db1 = (1.0 / static_cast<float>(N)) * getdB1(x[j], y[j], estimate);

		    estimate -> b1 = (estimate -> b1) - (STEP_SIZE_BATCH * db1);
  	}
  	return estimate;
}
