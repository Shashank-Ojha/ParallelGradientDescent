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

float getdB0(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction);
}

float getdB1(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction)*x;
}

float calculate_error(int N, float* x, float* y, estimate_t* estimate) {
	float res = 0;
	for (int i = 0; i < N; i++) {
    float est = evaluate(estimate, x[i]);
		res += (y[i] - est) * (y[i] - est);
	}

	return res / static_cast<float>(N);
}

estimate_t* bgd(int N, float* x, float* y, int num_threads)
{
	  omp_set_num_threads(num_threads);
    num_t* partial_db0 = (num_t*)malloc(sizeof(num_t) * num_threads);
		num_t* partial_db1 = (num_t*)malloc(sizeof(num_t) * num_threads);
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
    estimate -> b0 = INIT_B0;
	  estimate -> b1 = INIT_B1;

	  for(int i = 0; i < NUM_ITER_BATCH; i++)
		{
				for(int k = 0; k < num_threads; k++) {
          partial_db0[k].num = 0.0;
					partial_db1[k].num = 0.0;
				}

				int j, tid;

		    #pragma omp parallel for default(shared) private(j, tid) schedule(static)
			    for(j = 0; j < N; j++)
			    {
            float local_db0 = getdB0(x[j], y[j], estimate, N);
			      float local_db1 = getdB1(x[j], y[j], estimate, N);

				  	tid = omp_get_thread_num();
					  // TODO: change the partial sum arrays so they have padding
					  // 		right now, only a float so there is false sharing and so it is slower
            partial_db0[tid].num += local_db0;
					  partial_db1[tid].num += local_db1;
			  	}

        float db0 = 0.0;
        float db1 = 0.0;
				for (int k = 0; k < num_threads; k++)
				{
          db0 += partial_db0[k].num;
					db1 += partial_db1[k].num;
				}

        estimate -> b0 -= (STEP_SIZE_BATCH * db0);
		    estimate -> b1 -= (STEP_SIZE_BATCH * db1);
  	}
  	return estimate;
}

estimate_t* sgd_step(int N, float* x, float* y, estimate_t* estimate)
{
    int j = rand() % N;

    float db0 = getdB0(x[j], y[j], estimate, N);
    float db1 = getdB1(x[j], y[j], estimate, N);

    estimate -> b0 -= (STEP_SIZE_STOCH * db0);
    estimate -> b1 -= (STEP_SIZE_STOCH * db1);

  	return estimate;
}

estimate_t* sgd(int N, float* x, float* y)
{
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b0 = INIT_B0;
    estimate -> b1 = INIT_B1;

	  for(int i = 0; i < NUM_ITER_STOCH; i++)
		{
        estimate = sgd_step(N, x, y, estimate);
  	}
  	return estimate;
}

/*
        This is just doing an alpha-approx using SGD on
        a single thread
 */
estimate_t* sgd_approx(int N, float* x, float* y, float alpha, float refMSE, double* time)
{
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
    estimate -> b0 = INIT_B0;
    estimate -> b1 = INIT_B1;

    int num_steps = 0;
    while(true)
    {
      auto start = Clock::now();
      *estimate = *sgd_step(N, x, y, estimate);
      auto end = Clock::now();
      *time += duration_cast<dsec>(end - start).count();


      float MSE = calculate_error(N, x, y, estimate);
      float std_error = abs(MSE - refMSE) / sqrt(refMSE);
      if(num_steps > ITER_LIMIT || (std_error < alpha))
        break;

      num_steps++;
    }

    printf("num_steps for sequential: %d\n", num_steps);

  	return estimate;
}
