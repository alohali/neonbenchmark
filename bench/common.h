#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cassert>


void build_pointer_chain(void *p, size_t stride, size_t length) {
  
  size_t num = length / stride;
  for (size_t i = 0; i < num; ++i) {
    size_t dst = (i == (num - 1))?  reinterpret_cast<size_t>(p) : reinterpret_cast<size_t>(p) + (i + 1) * stride;
    size_t* src = reinterpret_cast<size_t*>(reinterpret_cast<size_t>(p) + i * stride);
    *src = dst;
  }
}


void print_pointer_chain(void *p, size_t stride, size_t length) {

  size_t num = length / stride;
  for (size_t i = 0; i < num; ++i) {
    printf("%x, %x\n", reinterpret_cast<size_t>(p), *(reinterpret_cast<size_t*>(p)));
  }
}


void ldr_to_use_pattern(void *p, size_t loop) {
  assert((loop % 64) == 0); 
  __asm__ __volatile__ (
    ".align 2\n"
    "1:\n"
    ".rept 64\n"
    "ldr %0, [%0]\n"
    ".endr 64\n"
    "subs %1, %1, #1\n"
    "bne 1b\n"
    :
    :"r"(p),"r"(loop / 64)
    :"cc"
  );
}

void ldr_bw(void *p, size_t length, size_t stride, size_t loop) {
  size_t iteration = length / stride;
  assert((iteration % 32) == 0);
  for (size_t l = 0; l < loop; ++l) { 
    void *temp_p = p;
    __asm__ __volatile__ (
      "mov r0, %0\n"
      ".align 2\n"
      "1:\n"
      ".rept 4\n"
      "vld1.f32 {d0-d1}, [r0], %2\n"
      "vld1.f32 {d2-d3}, [r0], %2\n"
      "vld1.f32 {d4-d5}, [r0], %2\n"
      "vld1.f32 {d6-d7}, [r0], %2\n"
      "vld1.f32 {d8-d9}, [r0], %2\n"
      "vld1.f32 {d10-d11}, [r0], %2\n"
      "vld1.f32 {d12-d13}, [r0], %2\n"
      "vld1.f32 {d14-d15}, [r0], %2\n"
      ".endr\n"
      "subs %1, %1, #1\n"
      "bne 1b\n"
      :
      :"r"(temp_p),"r"(iteration / 32), "r"(stride)
      :"cc","r0","r1","r2","r3","r4","q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11","q12","q13","q14","q15"
    );
  }
}

void inst_bw(void *p, size_t length,  size_t loop) {
  size_t iteration = length;
  assert((iteration % 32) == 0);
  for (size_t l = 0; l < loop; ++l) { 
    void *temp_p = p;
    __asm__ __volatile__ (
      "mov r0, %0\n"
      ".align 2\n"
      "1:\n"
      ".rept 4\n"
      "vabs.f32 q0,q0\n"
      "vabs.f32 q1,q1\n"
      "vabs.f32 q2,q2\n"
      "vabs.f32 q3,q3\n"
      "vabs.f32 q5,q5\n"
      "vabs.f32 q6,q6\n"
      "vabs.f32 q7,q7\n"
      "vabs.f32 q8,q8\n"
      ".endr\n"
      "subs %1, %1, #1\n"
      "bne 1b\n"
      :
      :"r"(temp_p),"r"(iteration / 32)
      :"cc","r0","r1","r2","r3","r4","q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11","q12","q13","q14","q15"
    );
  }
}
void str_bw(void *p, size_t length, size_t stride, size_t loop) {
  size_t iteration = length / stride;
  assert((iteration % 32) == 0);
  for (size_t l = 0; l < loop; ++l) { 
    void *temp_p = p;
    __asm__ __volatile__ (
      "mov r0, %0\n"
      ".align 2\n"
      "1:\n"
      ".rept 4\n"
      "vst1.f32 {d0-d1}, [r0], %2\n"
      "vst1.f32 {d2-d3}, [r0], %2\n"
      "vst1.f32 {d4-d5}, [r0], %2\n"
      "vst1.f32 {d6-d7}, [r0], %2\n"
      "vst1.f32 {d8-d9}, [r0], %2\n"
      "vst1.f32 {d10-d11}, [r0], %2\n"
      "vst1.f32 {d12-d13}, [r0], %2\n"
      "vst1.f32 {d14-d15}, [r0], %2\n"
      ".endr\n"
      "subs %1, %1, #1\n"
      "bne 1b\n"
      :
      :"r"(temp_p),"r"(iteration / 32), "r"(stride)
      :"cc","r0","r1","r2","r3","r4","q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11","q12","q13","q14","q15"
    );
  }
}

void copy_intrin(void *dst, void *src, size_t length, size_t stride, size_t loop) {

  char *loop_dst = (char *)dst;
  char *loop_src = (char *)src;
  size_t iteration = length / stride;
  assert((iteration % 4) == 0);
  for (size_t l = 0; l < loop; ++l) { 
     for(size_t i=0; i<iteration; i++){
       loop_dst[i] = loop_src[i];
     }    
  }
}
void copy_bw(void *dst, void *src, size_t length, size_t stride, size_t loop) {
  size_t iteration = length / stride;
  assert((iteration % 4) == 0);
  for (size_t l = 0; l < loop; ++l) { 
    __asm__ __volatile__ (
      ".align 2\n"
      "1:\n"

      "vld1.f32 {d0-d1}, [%1], %3\n"
      "vld1.f32 {d2-d3}, [%1], %3\n"
      "vst1.f32 {d4-d5}, [%0], %3\n"
      "vst1.f32 {d6-d7}, [%0], %3\n"

      "subs %2, %2, #1\n"
      "bne 1b\n"
      :
      :"r"(dst),"r"(src),"r"(iteration / 2), "r"(stride)
      :"cc","r0","r1","r2","r3","r4","q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11","q12","q13","q14","q15"
    );
  }
}

void add_in_place_bw(void *p, size_t length, size_t stride, size_t loop) {
  size_t iteration = length / stride;
  assert((iteration % 8) == 0);
  for (size_t l = 0; l < loop; ++l) { 
    __asm__ __volatile__ (
      "mov r0, %0\n"
      "mov r1, %0\n"
      ".align 2\n"
      "1:\n"
      "ldr r0, [%0]\n"
      "vld1.f32 {d0-d1}, [r0], %2\n"
      "vld1.f32 {d2-d3}, [r0], %2\n"
      "vadd.f32 q0, q0, q15\n"
      "vld1.f32 {d4-d5}, [r0], %2\n"
      "vadd.f32 q1, q1, q15\n"
      "vld1.f32 {d6-d7}, [r0], %2\n"
      "vadd.f32 q2, q2, q15\n"
      "vst1.f32 {d0-d1}, [r1], %2\n"
      "vld1.f32 {d8-d9}, [r0], %2\n"
      "vadd.f32 q3, q3, q15\n"
      "vst1.f32 {d2-d3}, [r1], %2\n"
      "vld1.f32 {d10-d11}, [r0], %2\n"
      "vadd.f32 q4, q4, q15\n"
      "vst1.f32 {d4-d5}, [r1], %2\n"
      "vld1.f32 {d12-d13}, [r0], %2\n"
      "vadd.f32 q5, q5, q15\n"
      "vst1.f32 {d6-d7}, [r1], %2\n"
      "vld1.f32 {d14-d15}, [r0], %2\n"
      "vadd.f32 q6, q6, q15\n"
      "vst1.f32 {d8-d9}, [r1], %2\n"
      "vadd.f32 q7, q7, q15\n"
      "vst1.f32 {d10-d11}, [r1], %2\n"
      "vst1.f32 {d12-d13}, [r1], %2\n"
      "vst1.f32 {d14-d15}, [r1], %2\n"
      "subs %1, %1, #1\n"
      "bne 1b\n"
      :
      :"r"(p),"r"(iteration / 8), "r"(stride)
      :"cc","r0","r1","r2","r3","r4","q0","q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","q11","q12","q13","q14","q15"
    );
  }
}
