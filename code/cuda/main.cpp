#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "regression.h"

estimate_t* bgdCuda(int N, float* x, float* y);
estimate_t* sgdCuda(int N, float* x, float* y, float alpha, float opt,
                    int blocks, int threadsPerBlock);

void printCudaInfo();

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  Filename\n");
    printf("  -a  Alpha\n");
    printf("  -b  Number of blocks\n");
    printf("  -t  Threads per block\n");
    printf("  -?  This message\n");
}

int checkInputArguments(char* filename, float alpha, int blocks,
                        int threadsPerBlock)
{
    if(filename == NULL)
    {
        printf("No input file given\n");
        return -1;
    }

    if(alpha == -1)
    {
        printf("Alpha was not specified\n");
        return -1;
    }

    if(blocks == -1)
    {
        printf("Number of blocks was not specified\n");
        return -1;
    }

    if(threadsPerBlock == -1)
    {
        printf("Threads per block was not specified\n");
        return -1;
    }

    return 0;
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
    float alpha = -1;
    int blocks = -1;
    int threadsPerBlock = -1;


    //Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:a:b:t:")) != EOF) {
        switch (opt) {
        case 'f':
            filename = optarg;
            break;
        case 'a':
            alpha = (float)atof(optarg);
            break;
        case 'b':
            blocks = atoi(optarg);
            break;
        case 't':
            threadsPerBlock = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if(checkInputArguments(filename, alpha, blocks, threadsPerBlock) == -1){
      usage(argv[0]);
      return 1;
    }

    // Read File
    int N;

    float refSlope;
    float refStdDev;
    float refMSE;

    float* x;
    float* y;

    FILE *input = fopen(filename, "r");
    if (!input) {
      printf("Unable to open file: %s.\n", filename);
      return 1; //Error
    }

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

    srand(418);

    estimate_t* estimate_bgd = bgd(N, x, y);
    printf("Batch: y = %.2f (x)\n", estimate_bgd -> b1);

    estimate_t* estimate_sgd = sgd(N, x, y);
    printf("Stochastic: y = %.2f (x)\n", estimate_sgd -> b1);

    estimate_t* estimate_sgdCuda = sgdCuda(N, x, y, alpha, refSlope,
                                           blocks, threadsPerBlock);
                                           
    printf("Cuda Stochastic: y = %.2f (x)\n", estimate_sgdCuda -> b1);

    return 0;
}
