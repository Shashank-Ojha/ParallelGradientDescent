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
			This design simply runs sgd on each thread and
			averages the results of b0 and b1 to compute the final answer
 */
estimate_t* sgd_design5(int N, float* x, float* y, int num_threads){

  estimate_t* estimates = (estimate_t*)malloc(sizeof(estimate_t) * num_threads);

  omp_set_num_threads(num_threads);
	int tid;

  // Run sgd in parallel to average the results
  #pragma omp parallel default(shared) private(tid)
	{
	  tid = omp_get_thread_num();
		estimates[tid] = *sgd(N, x, y);
  }

	estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));

  for(int j = 0; j < num_threads; j++) {
	  ret -> b0 += estimates[j].b0;
	  ret -> b1 += estimates[j].b1;
  }

  ret -> b0 /= static_cast<float>(num_threads);
  ret -> b1 /= static_cast<float>(num_threads);

  return ret;
}
