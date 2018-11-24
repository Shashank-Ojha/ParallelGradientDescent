typedef struct {
  float b0;
  float b1;
} estimate_t;

typedef struct {
  float num;
  char padding[60];
} num_t;

const float STEP_SIZE_BATCH = 0.0001;
const int NUM_ITER_BATCH =  1000;

const float STEP_SIZE_STOCH = 0.0001;
const int NUM_ITER_STOCH =  1000;
const int BATCH_SIZE_STOCH = 5;

const float INIT_B0 = 0.0;
const float INIT_B1 = 0.0;
