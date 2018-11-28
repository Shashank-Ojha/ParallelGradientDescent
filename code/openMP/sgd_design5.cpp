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
estimate_t* sgd_design5(int N, float* x, float* y, float alpha, float opt, int num_threads){

  estimate_t* estimates = (estimate_t*)malloc(sizeof(estimate_t) * num_threads);

	for(int t = 0; t < num_threads; t++)
	{
		estimates[t].b1 = INIT_B1;
	}

	estimate_t* ret = (estimate_t*)malloc(sizeof(estimate_t));
	ret -> b1 = 0.0;

  omp_set_num_threads(num_threads);
	int tid;

	float upper = opt + (alpha/2.0)*opt;
	float lower = opt - (alpha/2.0)*opt;

  // Run sgd in parallel to average the results

  #pragma omp parallel default(shared) private(tid)
	{
		tid = omp_get_thread_num();
		int num_steps = 0;
		while(true)
		{
			estimates[tid] = *sgd_step(N, x, y, estimates+tid);

			#pragma omp barrier

			//average the solutions
			if(tid == 0)
			{
				ret -> b1 = 0.0;
				for(int j = 0; j < num_threads; j++) {
					ret -> b1 += estimates[j].b1;
				}

				ret -> b1 /= static_cast<float>(num_threads);
			}

      #pragma omp barrier
      if(num_steps > ITER_LIMIT || (lower < (ret -> b1) && (ret -> b1) < upper))
  			break;

			num_steps++;
		}
		if(tid == 0){
			printf("num_steps: %d\n", num_steps);
		}
  }

  return ret;
}
