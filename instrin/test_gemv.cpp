#include <arm_neon.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <cmath>
#define L1BUFFER_SIZE 8192

void gemv_neon(float *a, float *b, float *c, int m, int k, float *l1buffer){
    float32x4_t vc[4];
    float32x4_t va[4];
    int maxbufferk = L1BUFFER_SIZE / 16 / 4;
    int ktile = k>maxbufferk? maxbufferk:k;
    for(int i=0; i<m; i+=16){
        vc[0] = vdupq_n_f32(0.);
        vc[1] = vdupq_n_f32(0.);
        vc[2] = vdupq_n_f32(0.);
        vc[3] = vdupq_n_f32(0.);
        for(int kt=0; kt<k; kt+=ktile){
            float *l1 = l1buffer;
            float32x4_t *tmp = (float32x4_t *)l1;
            for(int prek=0;prek<ktile;prek++){
                *tmp++ = vld1q_f32(a+i+0 +(prek+kt)*m);
                *tmp++ = vld1q_f32(a+i+4 +(prek+kt)*m);
                *tmp++ = vld1q_f32(a+i+8 +(prek+kt)*m);
                *tmp++ = vld1q_f32(a+i+12+(prek+kt)*m);
            }
            for(int j=0; j<ktile; j+=2){
                float32x2_t vb = vld1_f32(b+j+kt);
                for(int ji=0; ji<2; ji++){
                    va[0] = vld1q_f32(l1);l1+=4;
                    va[1] = vld1q_f32(l1);l1+=4;
                    va[2] = vld1q_f32(l1);l1+=4;
                    va[3] = vld1q_f32(l1);l1+=4;
                    vc[0] = vfmaq_lane_f32(vc[0], va[0], vb,ji);
                    vc[1] = vfmaq_lane_f32(vc[1], va[1], vb,ji);
                    vc[2] = vfmaq_lane_f32(vc[2], va[2], vb,ji);
                    vc[3] = vfmaq_lane_f32(vc[3], va[3], vb,ji);
                }
            }
        }
        vst1q_f32(c+i+0 , vc[0]);
        vst1q_f32(c+i+4 , vc[1]);
        vst1q_f32(c+i+8 , vc[2]);
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
    float *l1buffer;
    posix_memalign(reinterpret_cast<void**>(&l1buffer) , 128, 8192);
    for (auto miter = rows.begin(); miter != rows.end(); ++miter) {
        for(auto kiter = cols.begin(); kiter != cols.end(); ++kiter){
            int m = *miter; 
            int k = *kiter;
            int loop_cnt = 8192*32/(m/32)/(k/256);
            size_t read_size = (m * k + m +k)/1024 * sizeof(float) * loop_cnt;
  
            float *srca, *srcb, *dst, *ref;
            posix_memalign(reinterpret_cast<void**>(&srca), 128, m * k * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&srcb), 128, k * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&dst) , 128, m * sizeof(float));
            posix_memalign(reinterpret_cast<void**>(&ref) , 128, m * sizeof(float));
            for(size_t i=0; i<m*k; i++)
                srca[i] = rand() % 16 / 16.0 - 0.5;
            for(size_t i=0; i<k; i++)
                srcb[i] = rand() % 32 / 32.0 - 0.5;
            gemv_c(srca, srcb, ref, m, k);
            //init l1buffer
            for(int i=0; i<8192/4; i++)
                l1buffer[i] = 0.0;
            auto start = std::chrono::high_resolution_clock::now();
            for(int loop=0; loop<loop_cnt; loop++)
                gemv_neon(srca, srcb, dst, m, k, l1buffer);
            auto end = std::chrono::high_resolution_clock::now();
            auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            for(int i=0; i<m; i++){
                if(std::fabs(dst[i] - ref[i])>0.001 || !std::isfinite(dst[i])){
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

