estimate_t* sgd_with_k_samples(int N, double* x, double* y, int samplesPerThread,
                               int num_threads, double* workTime, double* commTime);

estimate_t* sgd_per_thread(int N, double* x, double* y, int num_threads, double* time);

// estimate_t* sgd_with_data_partition(int N, double* x, double* y, int num_threads);
