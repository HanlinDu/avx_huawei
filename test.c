#include <stdio.h>

#if f64 == 1
	#define AVX_TYPE double
	#define MATRIX_STEP (AVX_SIZE >> 6)
#elif f64 == 0
	#define AVX_TYPE float
	#define MATRIX_STEP (AVX_SIZE >> 5)

#endif

int main() {
	printf("%d\n", MATRIX_STEP);
}
