float evaluate(estimate_t* estimate, float x);

float getdB0(float x, float y, estimate_t* estimate);

float getdB1(float x, float y, estimate_t* estimate);

float calculate_error(int N, float* x, float* y, estimate_t estimate);
