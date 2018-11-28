#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "regression.h"


estimate_t* bgdCudaCopy(int N, float* x, float* y);
estimate_t* bgdCuda(int N, float* x, float* y);
estimate_t* sgdCuda(int N, float* x, float* y, float alpha, float opt);
void printCudaInfo();

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  Filename\n");
    printf("  -?  This message\n");
}

float evaluate(estimate_t* estimate, float x){
  return (estimate->b1)*x;
}

float getdB1(float x, float y, estimate_t* estimate){
  float prediction = evaluate(estimate, x);
  return -2.0 * (y-prediction)*x;
}

estimate_t* bgd(int N, float* x, float* y){
  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b1 = INIT_B1;

  for(int i = 0; i < NUM_ITER_BATCH; i++){
    float db1 = 0;
    for(int j = 0; j < N; j++)
    {
      db1 += (1.0 / static_cast<float>(N)) * getdB1(x[j], y[j], estimate);
    }

    estimate -> b1 = (estimate -> b1) - (STEP_SIZE_BATCH * db1);
  }
  return estimate;
}


estimate_t* sgd(int N, float* x, float* y){
  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b1 = INIT_B1;
  printf("num iter: %d\n", NUM_ITER_STOCH);
  for(int i = 0; i < NUM_ITER_STOCH; i++){
    //pick a point randomly
    int pi = rand() % N;

    float db1 = (1.0 / static_cast<float>(N)) * getdB1(x[pi], y[pi], estimate);

    estimate -> b1 = (estimate -> b1) - (STEP_SIZE_STOCH * db1);
  }

  return estimate;
}

int main(int argc, char** argv)
{
    char *filename = NULL;

    //Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "?f:")) != EOF) {

        switch (opt) {
        case 'f':
            filename = optarg;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if(filename == NULL){
      printf("No input file given\n");
      return 1;
    }

    // Read File
    int N;
    float* x;
    float* y;

    FILE *input = fopen(filename, "r");
    if (!input) {
      printf("Unable to open file: %s.\n", filename);
      return 1; //Error
    }

    fscanf(input, "%d\n", &N);

    x = (float*)malloc(sizeof(float) * N);
    y = (float*)malloc(sizeof(float) * N);

    for(int i = 0; i < N; i++){
      fscanf(input, "%f %f\n", x+i, y+i);
    }

    fclose(input);

    srand(418);

    estimate_t* estimate_bgd = bgd(N, x, y);
    printf("Batch: y = %.2f (x)\n", estimate_bgd -> b1);

    estimate_t* estimate_sgd = sgd(N, x, y);
    printf("Stochastic: y = %.2f (x)\n", estimate_sgd -> b1);

    // estimate_t* estimate_bgdCuda = bgdCuda(N, x, y);
    // printf("Cuda Batch: y = %.2f (x)\n", estimate_bgdCuda -> b1);


    estimate_t* estimate_sgdCuda = sgdCuda(N, x, y, 0.01, 3.136258);
    printf("Cuda Stochastic: y = %.2f (x)\n", estimate_sgdCuda -> b1);

    return 0;
}
