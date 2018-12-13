typedef struct {
  float b0;
  float b1;
  float b2;
  float b3;
} estimate_t;

const float INIT_B0 = 0.0;
const float INIT_B1 = 0.0;
const float INIT_B2 = 0.0;
const float INIT_B3 = 0.0;

typedef struct {
  float num;
  char padding[60];
} num_t;

const float STEP_SIZE_BATCH = 0.0000001;
const int NUM_ITER_BATCH =  100000;

const float STEP_SIZE_STOCH = 0.0000001;
const int NUM_ITER_STOCH =  100000;
const int BATCH_SIZE_STOCH = 5;
const int ITER_LIMIT = 100000;
