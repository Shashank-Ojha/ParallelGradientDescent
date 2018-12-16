typedef struct {
  double b3;
  double b2;
  double b1;
  double b0;
  char padding[48];
} estimate_t;

const double INIT_B3 = 0.0;
const double INIT_B2 = 0.0;
const double INIT_B1 = 0.0;
const double INIT_B0 = 0.0;

double evaluate(estimate_t* estimate, double x);

double getdB3(double x, double y, estimate_t* estimate, int N);
double getdB2(double x, double y, estimate_t* estimate, int N);
double getdB1(double x, double y, estimate_t* estimate, int N);
double getdB0(double x, double y, estimate_t* estimate, int N);

double calculate_error(int N, double* x, double* y, estimate_t* estimate);

estimate_t* bgd(int N, double* x, double* y, int num_threads, double* times);

void sgd_step(int N, double* x, double* y, estimate_t* estimate, int j);

estimate_t* sgd(int N, double* x, double* y);

void shuffle(double* x, double* y, int N, unsigned int* tid_seed);
