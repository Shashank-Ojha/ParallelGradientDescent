estimate_t* sgd_with_k_samples(int N, float* x, float* y, int samplesPerThread,
                               int num_threads, double* time);

estimate_t* sgd_per_thread(int N, float* x, float* y, int num_threads, double* time);

// estimate_t* sgd_with_data_partition(int N, float* x, float* y, int num_threads);
