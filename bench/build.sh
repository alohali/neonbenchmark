#g++ benchmark_memory.cpp -fopenmp -o a.out -g -ggdb -O0 -std=c++11 -marm -mfpu=neon -std=c++11 -mfloat-abi=softfp -static
g++ benchmark_memory.cpp -fopenmp -ofast -o a.out -std=c++11 -marm -mfpu=neon -std=c++11 
