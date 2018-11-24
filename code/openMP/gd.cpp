/**
 * Parallel Gradient Descent via OpenMP
 * Shashank Ojha(shashano), Kylee Santos(ksantos)
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include <algorithm>
#include <climits>

#include "regression.h"
#include "mic.h"


#define BUFSIZE 1024

static int _argc;
static const char **_argv;

/* Starter code function, don't touch */
const char *get_option_string(const char *option_name,
			      const char *default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return _argv[i + 1];
  return default_value;
}

/* Starter code function, do not touch */
int get_option_int(const char *option_name, int default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return atoi(_argv[i + 1]);
  return default_value;
}

/* Starter code function, do not touch */
float get_option_float(const char *option_name, float default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return (float)atof(_argv[i + 1]);
  return default_value;
}

/* Starter code function, do not touch */
static void show_help(const char *program_path)
{
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
    printf("\t-p <SA_prob>\n");
    printf("\t-i <SA_iters>\n");
}

float evaluate(estimate_t* estimate, float x){
  return (estimate->b1)*x + (estimate->b0);
}


float getdB0(float x, float y, estimate_t* estimate){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction);
}

float getdB1(float x, float y, estimate_t* estimate){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x;
}

estimate_t* bgd(int N, float* x, float* y, int num_threads,
				 num_t* partial_db0, num_t* partial_db1){
  omp_set_num_threads(num_threads);
  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b0 = 0.0;
  estimate -> b1 = 0.0;

  for(int i = 0; i < NUM_ITER_BATCH; i++){
	for(int k = 0; k < num_threads; k++) {
		partial_db0[k].num = 0;
		partial_db1[k].num = 0;
	}

    float db0 = 0;
    float db1 = 0;
	int j, tid;


    #pragma omp parallel for default(shared) private(j, tid) schedule(static)
    for(j = 0; j < N; j++)
    {
      float local_db0 = (1.0 / static_cast<float>(N)) * getdB0(x[j], y[j], estimate);
      float local_db1 = (1.0 / static_cast<float>(N)) * getdB1(x[j], y[j], estimate);
	  tid = omp_get_thread_num();
	  // TODO: change the partial sum arrays so they have padding
	  // 		right now, only a float so there is false sharing and so it is slower
	  partial_db0[tid].num += local_db0;
	  partial_db1[tid].num += local_db1;
  }

	for (int k = 0; k < num_threads; k++) {
		db0 += partial_db0[k].num;
		db1 += partial_db1[k].num;
	}

    estimate -> b0 = (estimate -> b0) - (STEP_SIZE_BATCH * db0);
    estimate -> b1 = (estimate -> b1) - (STEP_SIZE_BATCH * db1);
  }
  return estimate;
}


estimate_t* sgd(int N, float* x, float* y, int num_threads,
				num_t* results_b0, num_t* results_b1){
  omp_set_num_threads(num_threads);
  estimate_t estimate;
  estimate.b0 = INIT_B0;
  estimate.b1 = INIT_B1;

  int j, tid;

  // Run sgd in parallel to average the results
  #pragma omp parallel for default(shared) private(j, tid, estimate) schedule(static)
  for(j = 0; j < num_threads; j++){
	  for(int i = 0; i < NUM_ITER_STOCH; i++){
	    //pick a point randomly
	    int pi = rand() % N;

	    float db0 = getdB0(x[pi], y[pi], &estimate);
	    float db1 = getdB1(x[pi], y[pi], &estimate);

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

int main(int argc, const char *argv[])
{
	   using namespace std::chrono;

     typedef std::chrono::high_resolution_clock Clock;
     typedef std::chrono::duration<double> dsec;
     _argc = argc - 1;
     _argv = argv + 1;

     /* You'll want to use these parameters in your algorithm */
     const char *input_filename = get_option_string("-f", NULL);

     int num_of_threads = get_option_int("-n", 1);

		 num_t* partial_sums_db0 = (num_t*)malloc(sizeof(num_t) * num_of_threads);
		 num_t* partial_sums_db1 = (num_t*)malloc(sizeof(num_t) * num_of_threads);
		 num_t* results_b0 = (num_t*)malloc(sizeof(num_t) * num_of_threads);
		 num_t* results_b1 = (num_t*)malloc(sizeof(num_t) * num_of_threads);

     int error = 0;

     if (input_filename == NULL) {
       printf("Error: You need to specify -f.\n");
       error = 1;
     }

     if (error) {
       show_help(argv[0]);
       return 1;
     }

     printf("Number of threads: %d\n", num_of_threads);
     printf("Input file: %s\n", input_filename);

     FILE *input = fopen(input_filename, "r");

     if (!input) {
       printf("Unable to open file: %s.\n", input_filename);
       return -1;
     }

     int N;
     float* x;
     float* y;

     fscanf(input, "%d\n", &N);

     x = (float*)malloc(sizeof(float) * N);
     y = (float*)malloc(sizeof(float) * N);

     for(int i = 0; i < N; i++){
       fscanf(input, "%f %f\n", x+i, y+i);
     }

     fclose(input);

     #ifdef RUN_MIC /* Use RUN_MIC to distinguish between the target of compilation */

       /* This pragma means we want the code in the following block be executed in
        * Xeon Phi.
        */
     #pragma offload target(mic) \
       inout(x: length(N) INOUT) \
       inout(y: length(N) INOUT) \
	     inout(partial_sums_db0: length(num_of_threads) INOUT) \
       inout(partial_sums_db1: length(num_of_threads) INOUT) \
	     inout(results_b0: length(num_of_threads) INOUT) \
	     inout(results_b1: length(num_of_threads) INOUT)
     #endif
        {
           srand(418);

		  	 	 auto batch_start = Clock::now();
	         double batch_time = 0;

           estimate_t* estimate_bgd = bgd(N, x, y, num_of_threads,
			  	 																partial_sums_db0,
																					partial_sums_db1);


           printf("Batch: y = %.2f (x) + %0.2f\n", estimate_bgd -> b1, estimate_bgd -> b0);

		  	   auto batch_end = Clock::now();
		       batch_time += duration_cast<dsec>(batch_end - batch_start).count();
	  	     printf("Computation Time BGD: %lf.\n", batch_time);

		  	   auto stochastic_start = Clock::now();
		  	   double stochastic_time = 0;

           estimate_t* estimate_sgd = sgd(N, x, y, num_of_threads,
						 															results_b0, results_b1);

           printf("Stochastic: y = %.2f (x) + %0.2f\n", estimate_sgd -> b1, estimate_sgd -> b0);

				   auto stochastic_end = Clock::now();
				   stochastic_time += duration_cast<dsec>(stochastic_end - stochastic_start).count();
			  	 printf("Computation Time SGD: %lf.\n", stochastic_time);
		     }


	  free(x);
	  free(y);
	  free(partial_sums_db0);
	  free(partial_sums_db1);
	  free(results_b0);
	  free(results_b1);

  return 0;
}
