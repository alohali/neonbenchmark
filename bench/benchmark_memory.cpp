#include "common.h"
#include <cstring>


void benchmark_memory_latency(size_t mhz_freq) {

  std::vector<size_t> buffer_size = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}; // unit is kb
  //std::vector<size_t> stride = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608}; // unit is b
  std::vector<size_t> stride = {64}; // unit is b

  for (auto iter_buffer_size = buffer_size.begin(); iter_buffer_size != buffer_size.end(); ++iter_buffer_size) {
    for (auto iter_stride = stride.begin(); iter_stride != stride.end(); ++iter_stride) {
      size_t local_buffer_size = *iter_buffer_size * 1024;
      size_t local_stride = *iter_stride;
      if (local_buffer_size >= local_stride) {
        size_t *p;
        posix_memalign(reinterpret_cast<void**>(&p), 64, local_buffer_size);
        build_pointer_chain(reinterpret_cast<void*>(p), local_stride, local_buffer_size);
        auto start = std::chrono::high_resolution_clock::now();
        size_t inst_num = 256 * 1024 * 1024;
        ldr_to_use_pattern(reinterpret_cast<void*>(p), inst_num);
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float cpi =  1.0 * diff.count() * mhz_freq / inst_num;
        std::cerr << cpi << ",";
        free(p);
      } else {
        std::cerr << ",";
      }
    }
    std::cerr << std::endl;
  }
}
void benchmark_inst_bw(size_t mhz_freq) {

  std::vector<size_t> buffer_size = {2, 4, 6, 8,12}; // unit is kb

  for (auto iter_buffer_size = buffer_size.begin(); iter_buffer_size != buffer_size.end(); ++iter_buffer_size) {
      size_t local_buffer_size = (*iter_buffer_size) * 1024;
      size_t *p;
      posix_memalign(reinterpret_cast<void**>(&p), 64, local_buffer_size);
      size_t inst_num = 1024 * 1024 * 512;
      auto start = std::chrono::high_resolution_clock::now();
      inst_bw(reinterpret_cast<void*>(p), local_buffer_size, inst_num / local_buffer_size);
      auto end = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      float gop_per_second = inst_num  / 1000.0 / (diff.count()) ;
      std::cerr << "buffer = " << 1.0 * local_buffer_size / 1024.0 << "KB; "<<"time:"<<diff.count()/1000.0<<  " ms " << gop_per_second << "Gops/s";
      free(p);
    std::cerr << std::endl;
  }
}


void benchmark_memory_ldr_bw(size_t mhz_freq) {

  std::vector<size_t> buffer_size = {2, 4, 6, 8,12, 16, 32,48, 64, 128, 256,384, 512,768, 1024, 2048, 4096, 8192, 16384, 32768}; // unit is kb
  std::vector<size_t> stride = {16}; // unit is b

  for (auto iter_buffer_size = buffer_size.begin(); iter_buffer_size != buffer_size.end(); ++iter_buffer_size) {
    for (auto iter_stride = stride.begin(); iter_stride != stride.end(); ++iter_stride) {
      size_t local_buffer_size = (*iter_buffer_size) * 1024;
      size_t local_stride = *iter_stride;
      size_t *p;
      posix_memalign(reinterpret_cast<void**>(&p), 64, local_buffer_size);
      for (size_t i = 0; i < local_buffer_size / sizeof(size_t); ++i) {
        p[i] = i;
      }
      size_t inst_num = 1024 * 1024 * 512;
      auto start = std::chrono::high_resolution_clock::now();
      ldr_bw(reinterpret_cast<void*>(p), local_buffer_size, local_stride, inst_num / (local_buffer_size / local_stride));
      auto end = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      float byte_per_cycle = 1.0 * local_stride * inst_num / (1.0 * diff.count() * mhz_freq);

      float gb_per_second = inst_num  / (diff.count()) * local_stride/ 1000.0;
      std::cerr << "buffer = " << 1.0 * local_buffer_size / 1024.0 << "KB; "<<"time:"<<diff.count()/1000.0<<  " ms " << gb_per_second << "GB/s";
      free(p);
    }
    std::cerr << std::endl;
  }
}

void benchmark_memory_str_bw(size_t mhz_freq) {

  std::vector<size_t> buffer_size = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}; // unit is kb
  std::vector<size_t> stride = {16}; // unit is b

  for (auto iter_buffer_size = buffer_size.begin(); iter_buffer_size != buffer_size.end(); ++iter_buffer_size) {
    for (auto iter_stride = stride.begin(); iter_stride != stride.end(); ++iter_stride) {
      size_t local_buffer_size = (*iter_buffer_size) * 1024;
      size_t local_stride = *iter_stride;
      size_t *p;
      posix_memalign(reinterpret_cast<void**>(&p), 64, local_buffer_size);
      for (size_t i = 0; i < local_buffer_size / sizeof(size_t); ++i) {
        p[i] = i;
      }
      size_t inst_num = 1024 * 1024 * 512;
      auto start = std::chrono::high_resolution_clock::now();
      str_bw(reinterpret_cast<void*>(p), local_buffer_size, local_stride, inst_num / (local_buffer_size / local_stride));
      auto end = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      float byte_per_cycle = 1.0 * local_stride * inst_num / (1.0 * diff.count() * mhz_freq);
      float gb_per_second = 1.0 * byte_per_cycle * mhz_freq / 1000.0;
      std::cerr << "buffer = " << 1.0 * local_buffer_size / 1024.0 << "KB; "<<  byte_per_cycle << " Byte/cycle; " << gb_per_second << "GB/s";
      free(p);
    }
    std::cerr << std::endl;
  }
}

void benchmark_memory_copy_bw(size_t mhz_freq) {

  std::vector<size_t> buffer_size = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}; // unit is kb
  std::vector<size_t> stride = {16}; // unit is b

  for (auto iter_buffer_size = buffer_size.begin(); iter_buffer_size != buffer_size.end(); ++iter_buffer_size) {
    for (auto iter_stride = stride.begin(); iter_stride != stride.end(); ++iter_stride) {
      size_t local_buffer_size = (*iter_buffer_size) * 1024;
      size_t local_stride = *iter_stride;
      size_t *p1;
      size_t *p2;
      posix_memalign(reinterpret_cast<void**>(&p1), 64, local_buffer_size);
      posix_memalign(reinterpret_cast<void**>(&p2), 64, local_buffer_size);
      for (size_t i = 0; i < local_buffer_size / sizeof(size_t); ++i) {
        p1[i] = rand() % 256;
        p2[i] = 0;
      }
      size_t copy_num = 1024 * 1024 * 512;
      auto start = std::chrono::high_resolution_clock::now();
      for(volatile int i=0; i<copy_num  / local_buffer_size * local_stride; i++)
          memcpy(p2, p1, local_buffer_size);
      //copy_bw(reinterpret_cast<void*>(p2), reinterpret_cast<void*>(p1), local_buffer_size, local_stride, copy_num / (local_buffer_size / local_stride));
      auto end = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      float byte_per_cycle = 1.0 * local_stride * copy_num / (1.0 * diff.count() * mhz_freq);
      float gb_per_second = 1.0 * byte_per_cycle * mhz_freq / 1000.0;
      std::cerr << "buffer = " << 1.0 * local_buffer_size / 1024.0 << "KB; "<<  byte_per_cycle << " Byte/cycle; " << gb_per_second << "GB/s";
      for(size_t i=0; i<local_buffer_size / sizeof(size_t); i++){
          if(p1[i]!=p2[i]){
              std::cerr<<"[ERR]copy, i= "<<i<<"s="<<p1[i]<<",d="<<p2[i]<<std::endl;
              break;
          }
      }
      free(p1);
      free(p2);
    }
    std::cerr << std::endl;
  }
}

void benchmark_memory_add_inplace(size_t mhz_freq) {

  std::vector<size_t> buffer_size = {4, 8, 16, 32,64, 128, 256,1024,2048,4096,8196,16384}; // unit is kb
  std::vector<size_t> stride = {16}; // unit is b

  for (auto iter_buffer_size = buffer_size.begin(); iter_buffer_size != buffer_size.end(); ++iter_buffer_size) {
    for (auto iter_stride = stride.begin(); iter_stride != stride.end(); ++iter_stride) {
      size_t local_buffer_size = (*iter_buffer_size) * 1024;
      size_t local_stride = *iter_stride;
      size_t *p;
      posix_memalign(reinterpret_cast<void**>(&p), 64, local_buffer_size);
      for (size_t i = 0; i < local_buffer_size / sizeof(size_t); ++i) {
        p[i] = i;
      }
      size_t inst_num = 1024 * 1024 * 512;
      auto start = std::chrono::high_resolution_clock::now();
      add_in_place_bw(reinterpret_cast<void*>(p), local_buffer_size, local_stride, inst_num / (local_buffer_size / local_stride));
      auto end = std::chrono::high_resolution_clock::now();
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      float byte_per_cycle = 1.0 * local_stride * inst_num / (1.0 * diff.count() * mhz_freq);
      float gb_per_second = 1.0 * byte_per_cycle * mhz_freq / 1000.0;
      std::cerr << "buffer = " << 1.0 * local_buffer_size / 1024.0 << "KB; "<< diff.count()/1000.0 <<" ms,"<< byte_per_cycle << " Byte/cycle; " << gb_per_second << "GB/s";
      free(p);
    }
    std::cerr << std::endl;
  }
}

int main(int argc, char* argv[]) {

  size_t mhz_freq = 1843;//atoi(argv[1]);
//  printf("load:\n");
//  benchmark_memory_ldr_bw(mhz_freq);
//  printf("inst:\n");
//  benchmark_inst_bw(mhz_freq);
//printf("store:\n");
//  benchmark_memory_str_bw(mhz_freq);
//printf("copy:\n");
//    benchmark_memory_copy_bw(mhz_freq);
//printf("latency:\n");
  benchmark_memory_latency(mhz_freq);
    //benchmark_memory_copy_intrin_bw(mhz_freq);
//  benchmark_memory_add_inplace(mhz_freq);
  return 0;
}

