#include <stdio.h>
#include <stdlib.h>
#include <cilk/cilk.h>
#include <sys/time.h>
#include <assert.h>
#include <immintrin.h> // SSE/AVX/AVX2/AVX-512 instructions accelerate

#ifndef ni
#define ni 2048
#endif

#ifndef THRESHOLD
#define THRESHOLD 32
#endif

#ifndef AVX_TYPE
#define AVX_TYPE double
#endif

#ifndef AVX_SIZE
#define AVX_SIZE 512
#endif

AVX_TYPE A[ni][ni];
AVX_TYPE B[ni][ni];
AVX_TYPE C[ni][ni];

void set_param_type() {
#if (AVX_TYPE == double) && (AVX_SIZE == 512)
#define PARAM_TYPE __m512d
#define LOAD_FUNC _mm512_load_pd
#define ADD_FUNC _mm512_fmadd_pd
#define STORE_FUNC _mm512_store_pd

#elif (AVX_TYPE == int) && (AVX_SIZE == 512)
#elif (AVX_TYPE == double) && (AVX_SIZE == 256)
#elif (AVX_TYPE == int) && (AVX_SIZE == 256)
#elif (AVX_TYPE == double) && (AVX_SIZE == 128)
#elif (AVX_TYPE == int) && (AVX_SIZE == 128)
#endif
}

float tdiff(struct timeval *start, struct timeval *end)
{
	return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

void mm_base(AVX_TYPE *restrict C, int n_C,
			 AVX_TYPE *restrict A, int n_A,
			 AVX_TYPE *restrict B, int n_B,
			 int n)
{
	int i, j, k;
	PARAM_TYPE c0, c1, c2, c3;
	for (i = 0; i < n; ++i)
		for (k = 0; k < n; k+=32)
		{
			c0 = LOAD_FUNC(&C[i * n_C + 0]);
			c1 = LOAD_FUNC(&C[i * n_C + 8]);
			c2 = LOAD_FUNC(&C[i * n_C + 16]);
			c3 = LOAD_FUNC(&C[i * n_C + 24]);
			for (j = 0; j < n; j+=32)
			{
				c0 = ADD_FUNC(LOAD_FUNC(&A[i * n_A + j]), LOAD_FUNC(&B[j * n_B + k]), c0);
				c1 = ADD_FUNC(LOAD_FUNC(&A[i * n_A + j]), LOAD_FUNC(&B[j * n_B + k + 8]), c1);
				c2 = ADD_FUNC(LOAD_FUNC(&A[i * n_A + j]), LOAD_FUNC(&B[j * n_B + k + 16]), c2);
				c3 = ADD_FUNC(LOAD_FUNC(&A[i * n_A + j]), LOAD_FUNC(&B[j * n_B + k + 24]), c3);
			}
			STORE_FUNC(&C[i * n_C + k], c0);
			STORE_FUNC(&C[i * n_C + k + 8], c1);
			STORE_FUNC(&C[i * n_C + k + 16], c2);
			STORE_FUNC(&C[i * n_C + k + 24], c3);
		}
}

void mm_dac(AVX_TYPE *restrict C, int n_C,
			AVX_TYPE *restrict A, int n_A,
			AVX_TYPE *restrict B, int n_B,
			int n)
{
	assert((n & (-n)) == n);
	if (n <= THRESHOLD)
		mm_base(C, n_C, A, n_A, B, n_B, n);
	else
	{
#define X(M, r, c) (M + (r * (n_##M) + c) * (n / 2))
		mm_dac(X(C, 0, 0), n_C, X(A, 0, 0), n_A, X(B, 0, 0), n_B, n / 2);
		mm_dac(X(C, 0, 1), n_C, X(A, 0, 0), n_A, X(B, 0, 1), n_B, n / 2);
		mm_dac(X(C, 1, 0), n_C, X(A, 1, 0), n_A, X(B, 0, 0), n_B, n / 2);
		mm_dac(X(C, 1, 1), n_C, X(A, 1, 0), n_A, X(B, 0, 1), n_B, n / 2);
		
		mm_dac(X(C, 0, 0), n_C, X(A, 0, 1), n_A, X(B, 1, 0), n_B, n / 2);
		mm_dac(X(C, 0, 1), n_C, X(A, 0, 1), n_A, X(B, 1, 1), n_B, n / 2);
		mm_dac(X(C, 1, 0), n_C, X(A, 1, 1), n_A, X(B, 1, 0), n_B, n / 2);
		mm_dac(X(C, 1, 1), n_C, X(A, 1, 1), n_A, X(B, 1, 1), n_B, n / 2);
		
	}
}

int main(int argc, const char *argv[])
{
	for (int i = 0; i < ni; ++i)
	{
		for (int j = 0; j < ni; ++j)
		{
			A[i][j] = (AVX_TYPE)rand() / (AVX_TYPE)RAND_MAX;
			B[i][j] = (AVX_TYPE)rand() / (AVX_TYPE)RAND_MAX;
			C[i][j] = 0;
		}
	}
	struct timeval start, end;
	gettimeofday(&start, NULL);
	mm_dac(*C, ni, *B, ni, *A, ni, ni);
	gettimeofday(&end, NULL);
	printf("%0.6f\n", tdiff(&start, &end));
	return 0;
}
