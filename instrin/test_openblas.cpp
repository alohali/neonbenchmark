#include <vector>  
#include <iostream>  
#include <chrono>

#ifdef __cplusplus
extern "C"
{
#endif
#include <cblas.h>
#ifdef __cplusplus
}
#endif

using namespace std;  
int main() {  
    const int M=4096+16;  
    const int N=4096+16;  
    const float alpha=1;  
    const float beta=0;  
    float *A;  
    posix_memalign(reinterpret_cast<void**>(&A), 128, M*N * sizeof(float));
    float B[N] = {0};  
    float C[M] = {0};  
    
        cblas_sgemv(CblasRowMajor, CblasTrans, M, N, alpha, A, N, B, 1, beta, C, 1);  
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i < 10000; i++)
        cblas_sgemv(CblasRowMajor, CblasTrans, M, N, alpha, A, N, B, 1, beta, C, 1);  
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    
    std::cout <<"time: "<<diff.count() /1000.0<<"ms"<<std::endl;
    delete []A;
}
