#include <arm_neon.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <cmath>


void gemv_neon(float *a, float *b, float *c, int m, int k){

    for(int i=0; i<m; i+=16){
      float *at = a + i;
      float *bt = b;
      float *ct = c + i;
      __asm__ __volatile__ (
        "mov     r1, %6                        \n"
        "veor    q9,  q9,  q9                  \n"
        "veor    q10, q10, q10                 \n"
        "veor    q11, q11, q11                 \n"
        "veor    q12, q12, q12                 \n"
        ".align  8                          \n"
        "0:                                 \n"
        "vld1.f32    d0,      [%1]!         \n"
        "pld         [%0, #64]         \n"
        "vld1.f32    {d2-d3},  [%0]!         \n"
        "vld1.f32    {d4-d5},  [%0]!         \n"
        "vld1.f32    {d6-d7},  [%0]!        \n"
        "vld1.f32    {d8-d9},  [%0], %7      \n"
        "pld         [%0, #64]         \n"
        "vld1.f32    {d10-d11},[%0]!         \n"
        "vld1.f32    {d12-d13},[%0]!         \n"
        "vld1.f32    {d14-d15},[%0]!         \n"
        "vld1.f32    {d16-d17},  [%0], %7         \n"

        "vmla.f32    q9,  q1, d0[0]    \n"
        "pld         [%0]         \n"
        "vmla.f32    q10, q2, d0[0]    \n"
        "pld         [%0, %8]         \n"
        "vmla.f32    q11, q3, d0[0]    \n"
        "pld         [%0, %9]         \n"
        "vmla.f32    q12, q4, d0[0]    \n"
        "pld         [%0, %10]         \n"
        "vmla.f32    q9,  q5, d0[1]    \n"
        "vmla.f32    q10, q6, d0[1]    \n"
        "subs        r1, #2             \n"
        "vmla.f32    q11,  q7, d0[1]    \n"
        "vmla.f32    q12,  q8, d0[1]    \n"
        "bne        0b                   \n"

        "vst1.f32    {d18-d19},  [%2]!      \n"
        "vst1.f32    {d20-d21},  [%2]!      \n"
        "vst1.f32    {d22-d23},  [%2]!      \n"
        "vst1.f32    {d24-d25},  [%2]      \n"

        :"=r"(at),   // %0
         "=r"(bt),  // %1
         "=r"(ct)  // %2
        :"0"(at),    // %
         "1"(bt),
         "2"(ct),
         "r"(k),
         "r"(m*4-48),   // %7
         "r"(m*4),   // %8
         "r"(m*8),   // %9
         "r"(m*12)   // %9
        : "cc", "memory", "r0","r1", "q0", "q1", "q2", "q3", "q4", "q5","q6", "q7", "q8", "q9", "q10", "q11","q12"
      );
    }
}

void gemv_neon_intrin(float *a, float *b, float *c, int m, int k){

    for(int i=0; i<m; i+=16){
        float32x4_t vb;
        float32x4_t va[4];
        float32x4_t vc[4] = {vdupq_n_f32(0.), vdupq_n_f32(0.),vdupq_n_f32(0.),vdupq_n_f32(0.)};
        for(int j=0; j<k; j++){
            vb = vdupq_n_f32(b[j]);
            va[0]  = vld1q_f32(a + i + j * m);
            va[1]  = vld1q_f32(a + i + 4 + j * m);
            va[2]  = vld1q_f32(a + i + 8 + j * m);
            va[3]  = vld1q_f32(a + i + 12+ j * m);
            vc[0] = vmlaq_f32(vc[0], va[0], vb);
            vc[1] = vmlaq_f32(vc[1], va[1], vb);
            vc[2] = vmlaq_f32(vc[2], va[2], vb);
            vc[3] = vmlaq_f32(vc[3], va[3], vb);
        }
        vst1q_f32(c+i,    vc[0]);
        vst1q_f32(c+i+4,  vc[1]);
        vst1q_f32(c+i+8,  vc[2]);
        vst1q_f32(c+i+12, vc[3]);
    }
}
void gemv_c(float *a, float *b, float *c, int m, int k){
    float val;
    for(int i=0; i<m; i++){
        val = 0;
        for(int j=0; j<k; j++){
            val += a[i + j * m] * b[j];
        }
        c[i] = val;
    }
}


void gemv_test() {

    std::vector<int> rows = {64+16, 128+16, 256+16, 512+16, 1024+16,2048+16}; 
    std::vector<int> cols = {256+16, 512+16, 1024+16};
    for (auto miter = rows.begin(); miter != rows.end(); ++miter) {
        for(auto kiter = cols.begin(); kiter != cols.end(); ++kiter){
            int m = *miter; 
            int k = *kiter;
            int loop_cnt = 8192*2/(m/16)/(k/256);
            float read_size = (float)(m * k + m +k)/1024.0 * sizeof(float) * loop_cnt;
  
            float *srca, *srcb, *dst, *ref;
            posix_memalign(reinterpret_cast<void**>(&srca), 128, m * k * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&srcb), 128, k * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&dst) , 128, m * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&ref) , 128, m * sizeof(float));
            for(size_t i=0; i<m*k; i++)
                srca[i] = rand() % 32 / 32.0 - 0.5;
            for(size_t i=0; i<k; i++)
                srcb[i] = rand() % 32 / 32.0 - 0.5;
            gemv_c(srca, srcb, ref, m, k);
            auto start = std::chrono::high_resolution_clock::now();
            for(int loop=0; loop<loop_cnt; loop++)
                gemv_neon(srca, srcb, dst, m, k);
            auto end = std::chrono::high_resolution_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            for(int i=0; i<m; i++){
                if(std::fabs(dst[i] - ref[i])>0.001){
                    std::cout <<"error:"<<i<<" dst:"<<dst[i]<<" ref:"<<ref[i]<<std::endl;
                    return;
                }
            }
            float GB_per_second = float( read_size)  / (diff.count()) ;
            std::cout <<"loop_cnt="<<loop_cnt<< ",m= " << m << ",k= "<< k <<",time: "<<diff.count() /1000.0<<"ms,  BW:"<<GB_per_second<<std::endl;
             
            free(srca);
            free(srcb);
            free(dst);
            free(ref);
        } 
  
    }
}


int main(int argc, char* argv[]) {
    gemv_test();
    return 0;
}

