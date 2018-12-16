#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <climits>

#include "gd.h"
#include "regression.h"

double evaluate(estimate_t* estimate, double x){
  return (estimate->b3)*x*x*x + (estimate->b2)*x*x + (estimate->b1)*x + estimate->b0;
}

double getdB3(double x, double y, estimate_t* estimate, int N){
  double prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x*x*x / static_cast<double>(N);
}

double getdB2(double x, double y, estimate_t* estimate, int N){
  double prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x*x / static_cast<double>(N);
}

double getdB1(double x, double y, estimate_t* estimate, int N){
  double prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x / static_cast<double>(N);
}

double getdB0(double x, double y, estimate_t* estimate, int N){
  double prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction) / static_cast<double>(N);
}

double calculate_error(int N, double* x, double* y, estimate_t* estimate) {
	double res = 0.0;
	for (int i = 0; i < N; i++) {
    double y_hat = evaluate(estimate, x[i]);
		res += ((y[i] - y_hat) * (y[i] - y_hat)) / static_cast<double>(N);
	}

	return res;
}

estimate_t* bgd(int N, double* x, double* y, int num_threads, double* time)
{
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;
    auto start = Clock::now();

	  omp_set_num_threads(num_threads);
    double* partial_db0 = (double*)calloc(sizeof(double), num_threads);
		double* partial_db1 = (double*)calloc(sizeof(double), num_threads);
    double* partial_db2 = (double*)calloc(sizeof(double), num_threads);
    double* partial_db3 = (double*)calloc(sizeof(double), num_threads);
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
    estimate -> b0 = INIT_B0;
	  estimate -> b1 = INIT_B1;
    estimate -> b2 = INIT_B2;
  	estimate -> b3 = INIT_B3;
    int idx = 0;
	  for(int num_steps = 1; num_steps <= NUM_ITER_BATCH; num_steps++)
		{
        for(int i = 0; i < num_threads; i++){
          partial_db0[i] = 0.0;
          partial_db1[i] = 0.0;
          partial_db2[i] = 0.0;
          partial_db3[i] = 0.0;
        }
				int j, tid;
		    #pragma omp parallel for default(shared) private(j, tid) schedule(static)
			    for(j = 0; j < N; j++)
			    {
				  	tid = omp_get_thread_num();

            partial_db0[tid] += getdB0(x[j], y[j], estimate, N) * STEP_SIZE_BATCH;
					  partial_db1[tid] += getdB1(x[j], y[j], estimate, N) * STEP_SIZE_BATCH;
            partial_db2[tid] += getdB2(x[j], y[j], estimate, N) * STEP_SIZE_BATCH;
          	partial_db3[tid] += getdB3(x[j], y[j], estimate, N) * STEP_SIZE_BATCH;
			  	}

        double db0 = 0.0;
        double db1 = 0.0;
        double db2 = 0.0;
        double db3 = 0.0;
				for (int k = 0; k < num_threads; k++)
				{
          db0 += partial_db0[k];
					db1 += partial_db1[k];
          db2 += partial_db2[k];
          db3 += partial_db3[k];
				}

        estimate -> b0 -= db0;
		    estimate -> b1 -= db1;
        estimate -> b2 -= db2;
    		estimate -> b3 -= db3;
  	}

    auto end = Clock::now();
    *time += duration_cast<dsec>(end - start).count();

  	return estimate;
}

void sgd_step(int N, double* x, double* y, estimate_t* estimate, int j)
{
    j = j % N;
    estimate -> b0 -= STEP_SIZE_STOCH * getdB0(x[j], y[j], estimate, N);
    estimate -> b1 -= STEP_SIZE_STOCH * getdB1(x[j], y[j], estimate, N);
    estimate -> b2 -= STEP_SIZE_STOCH * getdB2(x[j], y[j], estimate, N);
    estimate -> b3 -= STEP_SIZE_STOCH * getdB3(x[j], y[j], estimate, N);
}

estimate_t* sgd(int N, double* x, double* y)
{
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b0 = INIT_B0;
    estimate -> b1 = INIT_B1;
    estimate -> b2 = INIT_B2;
    estimate -> b3 = INIT_B3;

	  for(int i = 0; i < NUM_ITER_STOCH; i++){
        int pi = rand() % N;
        sgd_step(N, x, y, estimate, pi);
  	}
  	return estimate;
}

estimate_t* sgd_epochs(int N, double* x, double* y)
{
	  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
	  estimate -> b0 = INIT_B0;
    estimate -> b1 = INIT_B1;
    estimate -> b2 = INIT_B2;
    estimate -> b3 = INIT_B3;

	  for(int i = 0; i < NUM_EPOCHS; i++){
      for(int j = 0; j < N; j++){
        sgd_step(N, x, y, estimate, j);
      }
  	}
  	return estimate;
}

void shuffle(double* x, double* y, int N, unsigned int* tid_seed)
{
  for(int i = 0; i < N; i++){
    int j = rand_r(tid_seed) % N;

    double tempx = x[i];
    x[i] = x[j];
    x[j] = tempx;

    double tempy = y[i];
    y[i] = y[j];
    y[j] = tempy;
  }
}
