#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <climits>

#include "sgd_1.h"

estimate_t* sgd(int N, float* x, float* y, int num_threads,
				num_t* partial_db0, num_t* partial_db1){

  estimate_t *estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b0 = INIT_B0;
  estimate -> b1 = INIT_B1;

  int tid;

  omp_set_num_threads(num_threads);

  // Run sgd in parallel to average the results
  #pragma omp parallel default(shared) private(tid)
  {
	  tid = omp_get_thread_num();
	  for(int i = 0; i < NUM_ITER_STOCH; i++){
        for(int j = 0; j < num_threads; k++) {
      		partial_db0[j].num = 0;
      		partial_db1[j].num = 0;
      	}
		float db0 = 0;
		float db1 = 0;
		// pick BATCH_SIZE_STOCH random points and average them
		for (int iter = 0; iter < BATCH_SIZE_STOCH; iter++) {
			//pick a point randomly
			int pi = rand() % N;
			partial_db0[tid] += getdB0(x[pi], y[pi], &estimate);
		    partial_db1[tid] += getdB1(x[pi], y[pi], &estimate);
		}
        partial_db0[tid] = partial_db0[tid] / (static_cast<float>(BATCH_SIZE_STOCH));
        partial_db1[tid] = partial_db1[tid] / (static_cast<float>(BATCH_SIZE_STOCH));

        // barrier to wait for all threads to compute their mini batches
        #pragma omp barrier
        if (tid == 0) {
            float avg_b0 = 0;
            float avg_b1 = 0;
            for (int k = 0; k < num_threads; k++) {
                avg_b0 += partial_db0[k];
                avg_b1 += partial_db1[k];
            }
            estimate -> b0 -= STEP_SIZE_STOCH * avg_b0 / (static_cast<float>(num_threads));
            estimate -> b1 -= STEP_SIZE_STOCH * avg_b1 / (static_cast<float>(num_threads));
        }
        #pragma omp barrier
	  }
  }

  return estimate;
}
