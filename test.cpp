#include <unistd.h>
#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include "neon.h"


int main(){

  const int h = 1080, w = 1920;
  unsigned char *src = (unsigned char *)malloc(h*w*3/2);
  unsigned char *dst = (unsigned char *)malloc(h*w*3);
  unsigned char *dst2 = (unsigned char *)malloc(h*w*3);
  for(int i=0; i<w*h;i++){
	src[i] = rand()%255;
  }
  for(int i=0; i<w*h/2;i++){
	src[i+w*h] = rand() % 255;
  }
  struct timezone zone;
  struct timeval time1;
  struct timeval time2;
  const int loop = 100;
  nv12_to_bgr(dst, src, w, h);//, unsigned char const* nv21, int width, int height); 
  gettimeofday(&time1, &zone);
  for (int i = 0; i < loop;i++ ) {
	nv12_to_bgr(dst, src, w, h);//, unsigned char const* nv21, int width, int height); 
  }
  gettimeofday(&time2, &zone);
  float delta = (time2.tv_sec - time1.tv_sec) * 1000 + (time2.tv_usec - time1.tv_usec) / 1000 ;
  printf("time cost: %f ms\n", delta/loop);
  free(src);
  free(dst);
  free(dst2);
}


