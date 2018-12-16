/**
 * Parallel Gradient Descent via CUDA
 * Shashank Ojha(shashano), Kylee Santos(ksantos)
 */

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "regression.h"

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  Filename\n");
    printf("  -b  Number of blocks\n");
    printf("  -t  Threads per block\n");
    printf("  -s  Samples per thread\n");
    printf("  -?  This message\n");
}

int checkInputArguments(char* filename, int blocks,
                        int threadsPerBlock, int samplesPerThread)
{
    if(filename == NULL) {
        printf("No input file given\n");
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

void print_estimate(char* design, estimate_t* est) {
    printf("%s:\n", design);
    printf("y = (%.5f) x^3 + (%.5f) x^2 + (%.5f) x + (%.5f)\n",
           est -> b3, est -> b2, est -> b1, est -> b0);
}

void print_MSE(estimate_t* est, float* x, float* y, int N) {
    printf("MSE: %.5f\n", calculate_error(N, x, y, est));
}

void print_divider() {
    printf("\n---------------------\n\n");
}

int main(int argc, char** argv)
{
    char *filename = NULL;
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

    if(checkInputArguments(filename, blocks, threadsPerBlock, samplesPerThread) == -1){
      usage(argv[0]);
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

    char* design;
    print_divider();

    estimate_t* est_bgd = bgd(N, x, y);
    design = (char*)("Batch");
    print_estimate(design, est_bgd);
    print_MSE(est_bgd, x, y, N);
    print_divider();

    design = (char*)("Stochastic");
    estimate_t* est_sgd = sgd(N, x, y);
    print_estimate(design, est_sgd);
    print_MSE(est_sgd, x, y, N);
    print_divider();

    design = (char*)("Stochastic by Epoch");
    estimate_t* est_sgd_epoch = sgd_epoch(N, x, y);
    print_estimate(design, est_sgd_epoch);
    print_MSE(est_sgd_epoch, x, y, N);
    print_divider();

    design = (char*)("Cuda Stochastic Per Thread");
    estimate_t* est_sgdCuda = sgdPerThread(N, x, y, blocks, threadsPerBlock);
    print_estimate(design, est_sgdCuda);
    print_MSE(est_sgdCuda, x, y, N);
    print_divider();

    design = (char*)("Cuda Stochastic by Block");
    estimate_t* est_sgdByBlock = sgdCudaByBlock(
        N, x, y, samplesPerThread, blocks
    );
    print_estimate(design, est_sgdByBlock);
    print_MSE(est_sgdByBlock, x, y, N);
    print_divider();

    design = (char*)("Cuda Stochastic with Partition");
    estimate_t* est_sgdPartition = sgdCudaWithPartition(N, x, y, blocks, threadsPerBlock);
    print_estimate(design, est_sgdPartition);
    print_MSE(est_sgdPartition, x, y, N);
    print_divider();

    return 0;
}
