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
estimate_t* sgd_design5(int N, float* x, float* y, float alpha, float refMSE,
                        int num_threads, double* time){

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

  // Run sgd in parallel to average the results
  #pragma omp parallel default(shared) private(tid)
	{
		tid = omp_get_thread_num();
		int num_steps = 0;
		for (int i = 0; i < NUM_ITER_STOCH; i++) {
      auto start = Clock::now();
			estimates[tid] = *sgd_step(N, x, y, estimates+tid);
      auto end = Clock::now();
      times[tid] += duration_cast<dsec>(end - start).count();

			#pragma omp barrier

			//average the solutions
			if(tid == 0)
			{
        double max_time = 0.0;
        ret -> b0 = 0.0;
				ret -> b1 = 0.0;
                ret -> b2 = 0.0;
        				ret -> b3 = 0.0;
				for(int j = 0; j < num_threads; j++) {
          ret -> b0 += estimates[j].b0 / static_cast<float>(num_threads);
					ret -> b1 += estimates[j].b1 / static_cast<float>(num_threads);
                    ret -> b2 += estimates[j].b2 / static_cast<float>(num_threads);
          					ret -> b3 += estimates[j].b3 / static_cast<float>(num_threads);
          if(times[j] > max_time){
            max_time = times[j];
          }
				}

        *time += max_time;
			}


      #pragma omp barrier

      if (i == 25 || i == 50 || i == 100 || i == 250 || i == 500) {
          float MSE = calculate_error(N, x, y, ret);
          float std_error = abs(MSE - refMSE) / sqrt(refMSE);
          printf("Steps: %d\tMSE: %.3f\n", i, MSE);
      }
    }
  }

  return ret;
}
