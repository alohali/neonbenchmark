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
/*
    __asm__ __volatile__ (
      :
      :"r"(c+i),"r"(a+i),"r"(b), "r"(k)
      :"cc","r0","r1","r2","r3","q0","q1","q2","q3"
    );
    */
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

    std::vector<int> rows = {32, 64, 128, 256, 512, 1024,2048}; 
    std::vector<int> cols = {256, 512, 1024};
    for (auto miter = rows.begin(); miter != rows.end(); ++miter) {
        for(auto kiter = cols.begin(); kiter != cols.end(); ++kiter){
            int m = *miter; 
            int k = *kiter;
            int loop_cnt = 8192/(m/32)/(k/256);
            size_t read_size = (m * k + m +k)/1024 * sizeof(float) * loop_cnt;
  
            float *srca, *srcb, *dst, *ref;
            posix_memalign(reinterpret_cast<void**>(&srca), 128, m * k * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&srcb), 128, k * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&dst) , 128, m * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&ref) , 128, m * sizeof(float));
            for(size_t i=0; i<m*k; i++)
                srca[i] = i % 16 / 16.0 - 0.5;
            for(size_t i=0; i<k; i++)
                srcb[i] = i%2 ;//rand() % 32 / 32.0 - 0.5;
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
            std::cout << "m= " << m << ",k= "<< k <<",time: "<<diff.count() /1000.0<<"ms,  BW:"<<GB_per_second<<std::endl;
             
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

