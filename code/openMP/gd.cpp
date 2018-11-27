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

#include "mic.h"

#include "gd.h"
#include "linear_regression.h"
#include "sgd_designs.h"

#define BUFSIZE 1024

static int _argc;
static const char **_argv;

/* Starter code function, don't touch */
const char *get_option_string(const char *option_name, const char *default_value)
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

	  double batch_time, stochastic_time;
	  estimate_t estimate_bgd, estimate_sgd;

    #ifdef RUN_MIC /* Use RUN_MIC to distinguish between the target of compilation */

       /* This pragma means we want the code in the following block be executed in
        * Xeon Phi.
        */
    #pragma offload target(mic) \
       inout(x: length(N) INOUT) \
       inout(y: length(N) INOUT)
    #endif
      {
         srand(418);

		     auto batch_start = Clock::now();

         estimate_bgd = *bgd(N, x, y, num_of_threads);

				 auto batch_end = Clock::now();
		  	 batch_time = duration_cast<dsec>(batch_end - batch_start).count();

		  	 auto stochastic_start = Clock::now();

      	 estimate_sgd = *sgd_design5(N, x, y, num_of_threads);

		     auto stochastic_end = Clock::now();
		     stochastic_time = duration_cast<dsec>(stochastic_end - stochastic_start).count();
      }

		// 100
	  //float reference = 110.2574;

		// 1000
		//0.32549(x)  -  8.81014
		//float reference = 0.3301031;

		// 5000
		//0.59644(x)  -  24.37033
		float reference = 0.3439843;

		// 10000
		//0.90189(x)  -  41.86405
		//float reference = 0.3035303;


	  float bgd_error = calculate_error(N, x, y, estimate_bgd);
	  float sgd_error = calculate_error(N, x, y, estimate_sgd);
	  float bgd_precent_error = (bgd_error - reference) / reference;
	  float sgd_precent_error = (sgd_error - reference) / reference;

	  printf("Batch: y = %.2f (x)\n", estimate_bgd.b1);
	  printf("Batch MSE: %0.2f\tPrecent Error: %0.2f\n", bgd_error, bgd_precent_error);
	  printf("Computation Time BGD: %lf.\n\n", batch_time);

	  printf("Stochastic: y = %.2f (x)\n", estimate_sgd.b1);
	  printf("Stochastic MSE: %0.2f\tPrecent Error: %0.2f\n", sgd_error, sgd_precent_error);
	  printf("Computation Time SGD: %lf.\n", stochastic_time);

	  free(x);
	  free(y);

  	return 0;
}
