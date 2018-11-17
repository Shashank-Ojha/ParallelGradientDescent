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

estimate_t* bgd(int N, float* x, float* y){
  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b0 = 0.0;
  estimate -> b1 = 0.0;

  //pick a point randomly
  for(int i = 0; i < NUM_ITER; i++){

    float db0 = 0;
    float db1 = 0;
    #pragma omp parallel for default(shared) schedule(static, 15)
    for(int j = 0; j < N; j++)
    {
      float local_db0 = (1.0 / static_cast<float>(N)) * getdB0(x[j], y[j], estimate);
      float local_db1 = (1.0 / static_cast<float>(N)) * getdB1(x[j], y[j], estimate);

      #pragma omp critical
      {
        db0 += local_db0;
        db1 += local_db1;
      }
    }

    estimate -> b0 = (estimate -> b0) - (STEP_SIZE * db0);
    estimate -> b1 = (estimate -> b1) - (STEP_SIZE * db1);
  }
  return estimate;
}


estimate_t* sgd(int N, float* x, float* y){
  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b0 = INIT_B0;
  estimate -> b1 = INIT_B1;

  for(int i = 0; i < NUM_ITER; i++){
    //pick a point randomly
    int pi = rand() % N;

    float db0 = getdB0(x[pi], y[pi], estimate);
    float db1 = getdB1(x[pi], y[pi], estimate);

    estimate -> b0 = (estimate -> b0) - (STEP_SIZE * db0);
    estimate -> b1 = (estimate -> b1) - (STEP_SIZE * db1);
  }

  return estimate;
}

int main(int argc, const char *argv[])
{
     _argc = argc - 1;
     _argv = argv + 1;

//   /* You'll want to use these parameters in your algorithm */
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

          estimate_t* estimate_bgd = bgd(N, x, y);
          printf("Batch: y = %.2f (x) + %0.2f\n", estimate_bgd -> b1, estimate_bgd -> b0);

          estimate_t* estimate_sgd = sgd(N, x, y);
          printf("Stochastic: y = %.2f (x) + %0.2f\n", estimate_sgd -> b1, estimate_sgd -> b0);
        }

  return 0;
}
