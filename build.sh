g++  -O3  -Ofast neon.cpp test.cpp  -flax-vector-conversions  -funroll-loops -march=armv8-a+simd -mtune=cortex-a57
