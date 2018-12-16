/**
 * Parallel Gradient Descent via OpenMP
 * Shashank Ojha(shashano), Kylee Santos(ksantos)
 *
 *
 *    Different types of designs:
 *         batch : 1 thread
 *         batch : n threads
 *         sgdPerThread : 1 thread
 *         sgd_epochsPerThread : 1 thread
 *         sgdPerThread : n threads
 *         sgd_epochsPerThread : n threads
 *         sgd_with_k_samples : 1 sample, 1 threads (SAME AS sgdPerThread : 1 thread)
 *         sgd_with_k_samples : 1 sample, n threads
 *         sgd_with_k_samples : k samples, 1 thread
 *         sgd_with_k_samples : k samples, n threads
 *
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
#include "regression.h"
#include "sgd_designs.h"

static int _argc;
static const char **_argv;

const char *get_option_string(const char *option_name, const char *default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return _argv[i + 1];
  return default_value;
}

int get_option_int(const char *option_name, int default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return atoi(_argv[i + 1]);
  return default_value;
}

double get_option_double(const char *option_name, double default_value)
{
  for (int i = _argc - 2; i >= 0; i -= 2)
    if (strcmp(_argv[i], option_name) == 0)
      return (double)atof(_argv[i + 1]);
  return default_value;
}

static void show_help(const char *program_path)
{
    printf("Usage: %s OPTIONS\n", program_path);
    printf("\n");
    printf("OPTIONS:\n");
    printf("\t-f <input_filename> (required)\n");
    printf("\t-n <num_of_threads> (required)\n");
    printf("\t-s <samples_per_thread> (required)\n");
}

static void print_divider()
{
  printf("\n");
  printf("-------------------------------------\n");
  printf("\n");
}

static void print_estimate_information(const char *estimateName, estimate_t* estimate,
                                       double MSE, double time)
{
  printf("%s:\n", estimateName);
  printf("y = (%.5f) x^3 + (%.5f) x^2 + (%.5f) x + (%.5f)\n",
         estimate -> b3, estimate -> b2,
         estimate -> b1, estimate -> b0);
  printf("MSE: %0.2f\n", MSE);
  printf("Computation Time: %lf.\n", time);
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
    int samplesPerThread = get_option_int("-s", 1);

    if (input_filename == NULL) {
      //error
      show_help(argv[0]);
      return 1;
    }

    printf("Number of threads: %d\n", num_of_threads);
    printf("Samples per thread: %d\n", samplesPerThread);
    printf("Input file: %s\n", input_filename);

    print_divider();

    FILE *input = fopen(input_filename, "r");

    if (!input) {
     printf("Unable to open file: %s.\n", input_filename);
     return 1;
    }

    int N;
    fscanf(input, "%d\n", &N);

    double* x = (double*)malloc(sizeof(double) * N);
    double* y = (double*)malloc(sizeof(double) * N);

    for(int i = 0; i < N; i++){
       fscanf(input, "%lf %lf\n", x+i, y+i);
    }

    fclose(input);

	  double batch_time = 0.0;
    double k_samples_time = 0.0;
    double per_thread_time = 0.0;

	  estimate_t estimate_bgd, estimate_sgd_kSamples, estimate_sgd_per_thread;

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

         estimate_bgd = *bgd(N, x, y, num_of_threads, &batch_time);

         estimate_sgd_kSamples = *sgd_with_k_samples(N, x, y, samplesPerThread, num_of_threads, &k_samples_time);

      	 estimate_sgd_per_thread = *sgd_per_thread(N, x, y, num_of_threads, &per_thread_time);
      }

	  double bgd_MSE = calculate_error(N, x, y, &estimate_bgd);
    double sgd_kSamples_MSE = calculate_error(N, x, y, &estimate_sgd_kSamples);
	  double sgd_per_thread_MSE = calculate_error(N, x, y, &estimate_sgd_per_thread);

    print_estimate_information("Batch", &estimate_bgd, bgd_MSE, batch_time);

    print_divider();

    print_estimate_information("SGD K Samples", &estimate_sgd_kSamples,
                                sgd_kSamples_MSE, k_samples_time);

    print_divider();

    print_estimate_information("SGD Per Thread", &estimate_sgd_per_thread,
                                sgd_per_thread_MSE, per_thread_time);

    print_divider();

	  free(x);
	  free(y);

  	return 0;
}
