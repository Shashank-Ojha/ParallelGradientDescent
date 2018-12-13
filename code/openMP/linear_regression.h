typedef struct {
  float b3;
  float b2;
  float b1;
  float b0;
  char padding[48];
} estimate_t;

const float INIT_B3 = 0.0;
const float INIT_B2 = 0.0;
const float INIT_B1 = 0.0;
const float INIT_B0 = 0.0;

float evaluate(estimate_t* estimate, float x);

float getdB3(float x, float y, estimate_t* estimate, int N);
float getdB2(float x, float y, estimate_t* estimate, int N);
float getdB1(float x, float y, estimate_t* estimate, int N);
float getdB0(float x, float y, estimate_t* estimate, int N);

float calculate_error(int N, float* x, float* y, estimate_t* estimate);

estimate_t* bgd(int N, float* x, float* y, int num_threads);

void sgd_step(int N, float* x, float* y, estimate_t* estimate, int j);

estimate_t* sgd(int N, float* x, float* y);

void shuffle(float* x, float* y, int N);

bool in_print_range(int i);

estimate_t* sgd_approx(int N, float* x, float* y, float alpha, float refMSE, double* time);
