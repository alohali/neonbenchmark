#include <arm_neon.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cassert>

void memcpy_asm(void* dst,void* src, size_t len){
    assert(len%128 == 0);
    void *s= src;
    void *d= dst;
    __asm__ __volatile__ (
       ".align 2\n"
       "1:\n"
       "vld1.f32 {d0- d1},[%1]!\n"
       "vld1.f32 {d2- d3},[%1]!\n"
       "vld1.f32 {d4- d5},[%1]!\n"
       "vld1.f32 {d6- d7},[%1]!\n"

       "vld1.f32 {d8- d9},[%1]!\n"
       "vld1.f32 {d10- d11},[%1]!\n"
       "vld1.f32 {d12- d13},[%1]!\n"
       "vld1.f32 {d14- d15},[%1]!\n"

       "vst1.f32 {d0- d1},[%0]!\n"
       "vst1.f32 {d2- d3},[%0]!\n"
       "vst1.f32 {d4- d5},[%0]!\n"
       "vst1.f32 {d6- d7},[%0]!\n"
       "vst1.f32 {d8- d9},[%0]!\n"
       "vst1.f32 {d10- d11},[%0]!\n"
       "vst1.f32 {d12- d13},[%0]!\n"
       "vst1.f32 {d14- d15},[%0]!\n"
       "subs %2, %2, #1\n"
       "bne 1b\n"
       :
       :"r"(d), "r"(s), "r"(len/128)
       :"cc", "r0","r1", "r2", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
    );
}

void memcpy_intrin(void* dst,void* src, size_t len){
    float32x4_t data[8];
    float32x4_t *s = (float32x4_t *)src;
    float32x4_t *d = (float32x4_t *)dst;
    for(size_t cnt=0; cnt<len/sizeof(float32x4_t);cnt++){
        d[cnt] = s[cnt];
    }
}


void benchmark_bw() {

  std::vector<size_t> buffer_size = {2, 4,  8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}; // unit is kb

  for (auto iter_buffer_size = buffer_size.begin(); iter_buffer_size != buffer_size.end(); ++iter_buffer_size) {
      size_t local_buffer_size = (*iter_buffer_size) * 1024;
      char *src ,*dst, *dst_ref;
      posix_memalign(reinterpret_cast<void**>(&src), 64, local_buffer_size);
      posix_memalign(reinterpret_cast<void**>(&dst), 64, local_buffer_size);
      posix_memalign(reinterpret_cast<void**>(&dst_ref), 64, local_buffer_size);
      for(size_t i=0; i<local_buffer_size; i++)
          src[i] = rand() % 256;
      size_t copy_size = 1024 * 1024 * 512;
      auto start = std::chrono::high_resolution_clock::now();
      for(size_t cnt=0; cnt<copy_size/local_buffer_size; cnt++)
          memcpy(dst_ref, src, local_buffer_size);
      auto end = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      float gb_per_second0 = copy_size  / 1024.0 / (diff.count()) ;

      memcpy_asm(dst, src, local_buffer_size);

      start = std::chrono::high_resolution_clock::now();
      for(size_t cnt=0; cnt<copy_size/local_buffer_size; cnt++)
          memcpy_asm(dst, src, local_buffer_size);
      end = std::chrono::high_resolution_clock::now();
      diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      float gb_per_second1 = copy_size  / 1024.0 / (diff.count()) ;

      for(int i=0; i<local_buffer_size; i++){
          if(src[i]!=dst[i]){
              std::cout<<"ERROR:"<<i<<","<<int(src[i])<<","<<int(dst[i])<<std::endl;
              break;
          }
      }
      std::cout << "buffer = " << 1.0 * local_buffer_size / 1024.0 << "KB;memcpy: "<<gb_per_second0<<", own_copy:"<<gb_per_second1<<std::endl;
      free(src);
      free(dst);
      free(dst_ref);
  }
}


int main(int argc, char* argv[]) {

  benchmark_bw();
  return 0;
}

