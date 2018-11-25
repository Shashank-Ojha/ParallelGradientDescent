#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <climits>

#include "regression.h"

estimate_t* sgd_design2(int N, float* x, float* y, int num_threads,
				num_t* results_b0, num_t* results_b1){
  omp_set_num_threads(num_threads);
  estimate_t estimate;
  estimate.b0 = INIT_B0;
  estimate.b1 = INIT_B1;

  float *partition_x[num_threads];
  float *partition_y[num_threads];
  int partition_sizes[num_threads];

  for(int t = 0; t < num_threads; t++) {
	  partition_sizes[t] = 0;
  }

  for(int k = 0; k < N; k++) {
	  int index = rand() % num_threads;
	  partition_x[index] = x[k];
	  partition_y[index] = y[k];
	  partition_sizes[index]++;
  }

  int j, tid;

  // Run sgd in parallel to average the results
  #pragma omp parallel for default(shared) private(j, tid, estimate) schedule(static)
  for(j = 0; j < num_threads; j++){
	  for(int i = 0; i < NUM_ITER_STOCH; i++){
	    //pick a point randomly
	    int pi = rand() % partition_sizes[j];

	    float db0 = getdB0(partition_x[j][pi], partition_y[j][pi], &estimate);
	    float db1 = getdB1(partition_x[j][pi], partition_y[j][pi], &estimate);

		estimate.b0 = (estimate.b0) - (STEP_SIZE_STOCH * db0);
	    estimate.b1 = (estimate.b1) - (STEP_SIZE_STOCH * db1);
	  }
	  results_b0[j].num = estimate.b0;
	  results_b1[j].num = estimate.b1;
  }

  float avg_b0 = 0;
  float avg_b1 = 0;
  for(int j = 0; j < num_threads; j++) {
	  avg_b0 += results_b0[j].num;
	  avg_b1 += results_b1[j].num;
  }

  avg_b0 = avg_b0 / static_cast<float>(num_threads);
  avg_b1 = avg_b1 / static_cast<float>(num_threads);
  estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));
  ret -> b0 = avg_b0;
  ret -> b1 = avg_b1;

  return ret;
}
