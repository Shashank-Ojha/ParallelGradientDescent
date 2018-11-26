#include "regression.h"
#include "linear_regression.h"

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

float calculate_error(int N, float* x, float* y, estimate_t estimate) {
	float res = 0;
	float b0 = estimate.b0;
	float b1 = estimate.b1;
	for (int i = 0; i < N; i++) {
		res += (y[i] - b0 - x[i] * b1) * (y[i] - b0 - x[i] * b1) / static_cast<float>(N);
	}

	return res;
}
