#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <climits>

#include "regression.h"

estimate_t* sgd_design3(int N, float* x, float* y, int num_threads,
				num_t* results_b0, num_t* results_b1){
  omp_set_num_threads(num_threads);
  estimate_t estimate;
  estimate.b0 = INIT_B0;
  estimate.b1 = INIT_B1;

  int j, tid;

  // Run sgd in parallel to average the results
  #pragma omp parallel default(shared) private(j, tid, estimate)
  {
	  tid = omp_get_thread_num();
	  for(int i = 0; i < NUM_ITER_STOCH; i++){
		float db0 = 0;
		float db1 = 0;
		// pick BATCH_SIZE_STOCH random points and average them before updating
		for (int j = 0; j < BATCH_SIZE_STOCH; j++) {
			//pick a point randomly
			int pi = rand() % N;
			db0 += getdB0(x[pi], y[pi], &estimate);
		    db1 += getdB1(x[pi], y[pi], &estimate);
		}

		estimate.b0 = (estimate.b0) - (STEP_SIZE_STOCH * db0 / (static_cast<float>(BATCH_SIZE_STOCH)));
	    estimate.b1 = (estimate.b1) - (STEP_SIZE_STOCH * db1 / (static_cast<float>(BATCH_SIZE_STOCH)));
	  }
	  results_b0[tid].num = estimate.b0;
	  results_b1[tid].num = estimate.b1;
  }

  float avg_b0 = 0;
  float avg_b1 = 0;
  for(int j = 0; j < num_threads; j++) {
	  avg_b0 += results_b0[j].num / static_cast<float>(num_threads);
	  avg_b1 += results_b1[j].num / static_cast<float>(num_threads);
  }

  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));
  ret -> b0 = avg_b0;
  ret -> b1 = avg_b1;

  return ret;
}
