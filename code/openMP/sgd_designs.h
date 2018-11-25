#include "regression.h"

estimate_t* sgd_design1(int N, float* x, float* y, int num_threads,
				num_t* results_b0, num_t* results_b1);
estimate_t* sgd_design2(int N, float* x, float* y, int num_threads,
				num_t* results_b0, num_t* results_b1);
estimate_t* sgd_design3(int N, float* x, float* y, int num_threads,
				num_t* results_b0, num_t* results_b1);
