typedef struct {
  float num;
  char padding[60];
} num_t;

const float STEP_SIZE_BATCH = 0.0000001;
const int NUM_ITER_BATCH =  100000;

const float STEP_SIZE_STOCH = 0.01;
const int NUM_ITER_STOCH =  100000;
const int BATCH_SIZE_STOCH = 5;
const int ITER_LIMIT = 10000000;
