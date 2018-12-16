typedef struct {
  float b0;
  float b1;
  float b2;
  float b3;
  char padding[48];
} estimate_t;

const float INIT_B3 = 0.0;
const float INIT_B2 = 0.0;
const float INIT_B1 = 0.0;
const float INIT_B0 = 0.0;

const float CLOCK_RATE = 1733000000.0;

const float STEP_SIZE_BATCH = 0.001;
const int NUM_ITER_BATCH =  10000;

const float STEP_SIZE_STOCH = 0.5;
const int NUM_ITER_STOCH = 250000;
const int NUM_ITER_STOCH_EPOCH = 50;
const int NUM_ITER_STOCH_BLOCK = 100000;
const int NUM_ITER_STOCH_PARTITION = 100;

void initialize_estimate(estimate_t* estimate);
float evaluate(estimate_t* estimate, float x);

float getdB3(float x, float y, estimate_t* estimate, int N);
float getdB2(float x, float y, estimate_t* estimate, int N);
float getdB1(float x, float y, estimate_t* estimate, int N);
float getdB0(float x, float y, estimate_t* estimate, int N);

float calculate_error(int N, float* x, float* y, estimate_t* estimate);

estimate_t* bgd(int N, float* x, float* y);
void sgd_step(int N, float* x, float* y, estimate_t* estimate, int j);
estimate_t* sgd(int N, float* x, float* y);
estimate_t* sgd_epoch(int N, float* x, float* y);

estimate_t* bgdCuda(int N, float* x, float* y);
estimate_t* sgdPerThread(int N, float* x, float* y, int blocks, int threadsPerBlock);
estimate_t* sgdCudaByBlock(int N, float* x, float* y, int samplesPerThread,
                           int blocks);

estimate_t* sgdCudaWithPartition(int N, float* x, float* y, int blocks,
                                 int threadsPerBlock);

void shuffle(float* x, float* y, int N, unsigned int* tid_seed);
