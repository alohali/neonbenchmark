# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
#include  <stdlib.h>
#include  <string.h>
#include  <xmmintrin.h>
#include "asm.h"
extern int omp_get_num_procs();
extern int omp_set_num_threads(int );
#define STREAM_ARRAY_SIZE (1024*1024*128)

float __attribute__ ((aligned (128))) src[STREAM_ARRAY_SIZE];
float __attribute__ ((aligned (128))) dst[STREAM_ARRAY_SIZE];

void tuned_STREAM_Copy()
{
	__m128 buf[8];
	// #pragma omp parallel for
    for (ssize_t j=0; j<STREAM_ARRAY_SIZE; j+=32){
    	buf[0] = _mm_load_ps(src + j + 0 * 4);
    	buf[1] = _mm_load_ps(src + j + 1 * 4);
    	buf[2] = _mm_load_ps(src + j + 2 * 4);
    	buf[3] = _mm_load_ps(src + j + 3 * 4);
    	buf[4] = _mm_load_ps(src + j + 4 * 4);
    	buf[5] = _mm_load_ps(src + j + 5 * 4);
    	buf[6] = _mm_load_ps(src + j + 6 * 4);
    	buf[7] = _mm_load_ps(src + j + 7 * 4);
    	_mm_store_ps(dst + j + 0 * 4, buf[0]);
    	_mm_store_ps(dst + j + 1 * 4, buf[1]);
    	_mm_store_ps(dst + j + 2 * 4, buf[2]);
    	_mm_store_ps(dst + j + 3 * 4, buf[3]);
    	_mm_store_ps(dst + j + 4 * 4, buf[4]);
    	_mm_store_ps(dst + j + 5 * 4, buf[5]);
    	_mm_store_ps(dst + j + 6 * 4, buf[6]);
    	_mm_store_ps(dst + j + 7 * 4, buf[7]);
    }
}

int main(int argc, char *argv[]){
	int loopcnt = 1;
	if(argc>1)
		loopcnt = atoi(argv[1]);
	memset(src, 0, sizeof(src));
	memset(dst, 0, sizeof(dst));
	printf("%lx %lx\n", (ssize_t)src, (ssize_t)dst);
	struct timeval start, end;
	struct timezone tz;
	tuned_STREAM_Copy();
	printf("num of thread: %d\n", omp_get_num_procs());
	omp_set_num_threads(12);
	gettimeofday(&start, &tz);
	for(int i=0; i<loopcnt; i++)
		tuned_STREAM_Copy();
	gettimeofday(&end, &tz);


	gettimeofday(&start, &tz);

	for(volatile int i=0; i<loopcnt; i++)
		fast_memcpy((uint8_t *)dst, (uint8_t *)src, sizeof(src));
	gettimeofday(&end, &tz);

	float ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
	printf("time %.3f ms, size %lu MB, bw %.3f GB/s\n", ms, sizeof(src) * 2 /1024 /1024, sizeof(src) * 2 /1024 /1024/ms * loopcnt );
}