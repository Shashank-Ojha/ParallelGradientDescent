#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "regression.h"

estimate_t* bgdCuda(int N, float* x, float* y);
estimate_t* sgdCuda(int N, float* x, float* y, float alpha, float opt,
                    int blocks, int threadsPerBlock);
estimate_t* sgdCudaByBlock(int N, float* x, float* y, float alpha, float opt,
                    int k, int blocks, int threadsPerBlock);

estimate_t* sgdCudaWithPartition(int N, float* x, float* y, float alpha, float opt,
                                 int blocks, int threadsPerBlock);

float calculateMSE(estimate_t* est, float* X, float* Y, int N);

float evaluate(estimate_t* estimate, float x);



void printCudaInfo();

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  Filename\n");
    printf("  -a  Alpha\n");
    printf("  -b  Number of blocks\n");
    printf("  -t  Threads per block\n");
    printf("  -s  Samples per thread\n");
    printf("  -?  This message\n");
}

int checkInputArguments(char* filename, float alpha, int blocks,
                        int threadsPerBlock, int samplesPerThread)
{
    if(filename == NULL) {
        printf("No input file given\n");
        return -1;
    }

    if(alpha == -1) {
        printf("Alpha was not specified\n");
        return -1;
    }

    if(blocks == -1) {
        printf("Number of blocks was not specified\n");
        return -1;
    }

    if(threadsPerBlock == -1) {
        printf("Threads per block was not specified\n");
        return -1;
    }

    if(samplesPerThread == -1) {
        printf("Samples per thread was not specified\n");
        return -1;
    }

    return 0;
}


float getdB0(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction);
}

float getdB1(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction)*x;
}

float getdB2(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction)*x*x;
}

float getdB3(float x, float y, estimate_t* estimate, int N){
  float prediction = evaluate(estimate, x);
  return (-2.0 / static_cast<float>(N)) * (y-prediction)*x*x;
}

estimate_t* bgd(int N, float* x, float* y){
  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b0 = INIT_B0;
  estimate -> b1 = INIT_B1;
  estimate -> b2 = INIT_B2;
  estimate -> b3 = INIT_B3;

  for(int i = 0; i < NUM_ITER_BATCH; i++){
    float db0 = 0.0;
    float db1 = 0.0;
    float db2 = 0.0;
    float db3 = 0.0;
    for(int j = 0; j < N; j++) {
      db0 += getdB0(x[j], y[j], estimate, N);
      db1 += getdB1(x[j], y[j], estimate, N);
      db2 += getdB2(x[j], y[j], estimate, N);
      db3 += getdB3(x[j], y[j], estimate, N);
    }

    estimate -> b0 -= STEP_SIZE_BATCH * db0;
    estimate -> b1 -= STEP_SIZE_BATCH * db1;
    estimate -> b2 -= STEP_SIZE_BATCH * db2;
    estimate -> b3 -= STEP_SIZE_BATCH * db3;
  }

  return estimate;
}

estimate_t* sgd(int N, float* x, float* y){
  estimate_t* estimate = (estimate_t*)malloc(sizeof(estimate_t));
  estimate -> b0 = INIT_B0;
  estimate -> b1 = INIT_B1;
  estimate -> b2 = INIT_B2;
  estimate -> b3 = INIT_B3;
  for(int i = 0; i < NUM_ITER_STOCH; i++){
    //pick a point randomly
    int pi = rand() % N;

    estimate -> b0 -= STEP_SIZE_STOCH * getdB0(x[pi], y[pi], estimate, N);
    estimate -> b1 -= STEP_SIZE_STOCH * getdB1(x[pi], y[pi], estimate, N);
    estimate -> b2 -= STEP_SIZE_STOCH * getdB2(x[pi], y[pi], estimate, N);
    estimate -> b3 -= STEP_SIZE_STOCH * getdB3(x[pi], y[pi], estimate, N);

    if (i == 25 || i == 100 || i == 250 || i == 500 || i == 1000 || i == 1500 ||
        i == 2000 || i == 2500 || i == 5000) {
        float MSE = calculateMSE(estimate, x, y, N);
        printf("Steps: %d\tMSE: %.3f\n", i, MSE);
    }
  }

  return estimate;
}

int main(int argc, char** argv)
{
    char *filename = NULL;
    float alpha = -1;
    int blocks = -1;
    int threadsPerBlock = -1;
    int samplesPerThread = -1;


    //Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:a:b:t:s:")) != EOF) {
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
        case 's':
            samplesPerThread = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if(checkInputArguments(filename, alpha, blocks, threadsPerBlock, samplesPerThread) == -1){
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
    printf("Batch: y = (%.3f) x^3 + (%.3f) x^2 + (%.3f) x + (%.3f)\n",
            estimate_bgd -> b3, estimate_bgd -> b2, estimate_bgd -> b1,
            estimate_bgd -> b0);

    estimate_t* estimate_sgd = sgd(N, x, y);
    printf("Stochastic: y = (%.3f) x^3 + (%.3f) x^2 + (%.3f) x + (%.3f)\n",
            estimate_sgd -> b3, estimate_sgd -> b2, estimate_sgd -> b1,
            estimate_sgd -> b0);

    estimate_t* estimate_sgdCuda = sgdCuda(N, x, y, alpha, refMSE,
                                           blocks, threadsPerBlock);

   printf("Cuda Stochastic: y = (%.3f) x^3 + (%.3f) x^2 + (%.3f) x + (%.3f)\n",
           estimate_sgdCuda -> b3, estimate_sgdCuda -> b2,
           estimate_sgdCuda -> b1, estimate_sgdCuda -> b0);



    // estimate_t* estimate_sgdCudaByBlock = sgdCudaByBlock(N, x, y, alpha,
    //                     refMSE, samplesPerThread, blocks, threadsPerBlock);
    //
    // printf("Cuda Stochastic by block: y = %.2f (x) + %.2f\n", estimate_sgdCudaByBlock -> b1, estimate_sgdCudaByBlock -> b0);

    estimate_t* estimate_sgdCudaWithPartition = sgdCudaWithPartition(N, x, y, alpha,
                        refMSE, blocks, threadsPerBlock);

    printf("Cuda Stochastic w Partition: y = (%.3f) x^3 + (%.3f) x^2 + (%.3f) x + (%.3f)\n",
            estimate_sgdCudaWithPartition -> b3,
            estimate_sgdCudaWithPartition -> b2,
            estimate_sgdCudaWithPartition -> b1,
            estimate_sgdCudaWithPartition -> b0);
    return 0;
}
