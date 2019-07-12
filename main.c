#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <assert.h>

#define ALIGN 64

void gather_smallbuf(
        double** restrict target,
        double*  const restrict source,
        int*     const restrict pat, //in spatter, this is a long. However, with the new small index buffers, int will do
        size_t pat_len,
        size_t delta,
        size_t n,
        size_t target_len) {
    #pragma omp parallel 
    {
        int t = omp_get_thread_num();

        #pragma omp for
        for (size_t i = 0; i < n; i++) {
           double *sl = source + delta * i;
           double *tl = target[t] + pat_len*(i%target_len);

           for (size_t j = 0; j < pat_len; j++) {
               tl[j] = sl[pat[j]];
           }
        }
    }
}

int main() {

    size_t pat_len = 8;
    size_t n = 1024;
    size_t ntargets = 1;

    size_t stride = 4;
    int nthreads = omp_get_num_threads();
    
    double **target = (double**)aligned_alloc(ALIGN, nthreads * sizeof(double*));
    long target_len = ntargets * pat_len;
    for (int i = 0; i < nthreads; i++) {
        target[i] = (double*)aligned_alloc(ALIGN, target_len * sizeof(double));
    }

    long source_len = n * stride * pat_len;
    double *source = (double*)aligned_alloc(ALIGN, source_len * sizeof(double));
    for (long i = 0; i < source_len; i++) {
        source[i] = i % 100;
    }

    int *pat = (int*)aligned_alloc(ALIGN, pat_len * sizeof(int));
    for (int i = 0; i < pat_len; i++) {
        pat[i] = stride*i;
    }

    gather_smallbuf(target, source, pat, pat_len, pat_len*stride, n, ntargets);

    // Make sure nothing is optimized out
    double tmp = 0;
    for (int i = 0; i < nthreads; i++) {
        for (int j = 0; j < target_len; j++) {
           tmp += target[i][j]; 
        }
    }
    assert(tmp!=0);

}

