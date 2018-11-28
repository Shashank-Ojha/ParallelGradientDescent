typedef struct {
  float b1;
  //padding
} estimate_t;

const float INIT_B1 = 0.0;

float evaluate(estimate_t* estimate, float x);

float getdB1(float x, float y, estimate_t* estimate);

float calculate_error(int N, float* x, float* y, estimate_t estimate);

estimate_t* bgd(int N, float* x, float* y, int num_threads);

estimate_t* sgd_step(int N, float* x, float* y, estimate_t* estimate);

estimate_t* sgd(int N, float* x, float* y);
