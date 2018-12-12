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
    printf("\t-a <alpha>\n");
}

static void print_divider()
{
  printf("\n");
  printf("-------------------------------------\n");
  printf("\n");
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
    float alpha = get_option_float("-a", 0.1);

    int error = 0;

    if (input_filename == NULL) {
       error = 1;
    }

   	if (error) {
     	 show_help(argv[0]);
       return 1;
    }

    printf("Number of threads: %d\n", num_of_threads);
    printf("Input file: %s\n", input_filename);

    print_divider();

    FILE *input = fopen(input_filename, "r");

    if (!input) {
     printf("Unable to open file: %s.\n", input_filename);
     return -1;
    }

    int N;

    float refSlope;
    float refStdDev;
    float refMSE;

    float* x;
    float* y;

    fscanf(input, "%d\n", &N);
    fscanf(input, "%f\n", &refSlope);
    fscanf(input, "%f\n", &refStdDev);
    fscanf(input, "%f\n", &refMSE);

    x = (float*)malloc(sizeof(float) * N);
    y = (float*)malloc(sizeof(float) * N);

    for(int i = 0; i < N; i++){
       fscanf(input, "%f %f\n", x+i, y+i);
    }

    fclose(input);

	  double batch_time, stochastic_sequential_time, stochastic_parallel_time;
	  estimate_t estimate_bgd, estimate_sgd_sequential, estimate_sgd_parallel;

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

         estimate_sgd_sequential = *sgd_approx(N, x, y, alpha, refMSE, &stochastic_sequential_time);

      	 estimate_sgd_parallel = *sgd_design5(N, x, y, alpha, refMSE, num_of_threads, &stochastic_parallel_time);
      }


	  float bgd_MSE = calculate_error(N, x, y, &estimate_bgd);
    float sgd_MSE_sequential = calculate_error(N, x, y, &estimate_sgd_sequential);
	  float sgd_MSE_parallel = calculate_error(N, x, y, &estimate_sgd_parallel);

    printf("Reference: y = %.2f (x) + %.2f\n", refSlope, 5.0); //TODO: Change to actually read in the intercept

    print_divider();

    printf("Batch:\n");
    printf("y = (%.5f) x^3 + (%.5f) x^2 + (%.5f) x + (%.5f)\n",
                estimate_bgd.b3, estimate_bgd.b2,
                estimate_bgd.b1, estimate_bgd.b0);
	  printf("MSE: %0.2f\n", bgd_MSE);
	  printf("Computation Time: %lf.\n", batch_time);

    print_divider();

    printf("Stochastic Sequential:\n");
    printf("y = (%.5f) x^3 + (%.5f) x^2 + (%.5f) x + (%.5f)\n",
                estimate_sgd_sequential.b3, estimate_sgd_sequential.b2,
                estimate_sgd_sequential.b1, estimate_sgd_sequential.b0);
    printf("MSE: %0.2f\n", sgd_MSE_sequential);
	  printf("Computation Time: %lf.\n", stochastic_sequential_time);

    print_divider();

    printf("Stochastic Parallel:\n");
	  printf("y = (%.5f) x^3 + (%.5f) x^2 + (%.5f) x + (%.5f)\n",
                estimate_sgd_parallel.b3, estimate_sgd_parallel.b2,
                estimate_sgd_parallel.b1, estimate_sgd_parallel.b0);
	  printf("MSE: %0.2f\n", sgd_MSE_parallel);
	  printf("Computation Time: %lf.\n", stochastic_parallel_time);

    print_divider();

	  free(x);
	  free(y);

  	return 0;
}
