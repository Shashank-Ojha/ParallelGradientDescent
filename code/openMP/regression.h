typedef struct {
  float b0;
  float b1;
} estimate_t;

typedef struct {
  float num;
  char padding[124];
} num_t;

const float STEP_SIZE_BATCH = 0.00005;
const int NUM_ITER_BATCH =  1000000;
const float STEP_SIZE_STOCH = 0.0001;
const int NUM_ITER_STOCH =  500000;
const float INIT_B0 = 0.0;
const float INIT_B1 = 0.0;
