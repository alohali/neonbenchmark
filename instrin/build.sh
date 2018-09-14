g++    -O2 test_gemv.cpp  -o gemv -flax-vector-conversions  -funroll-loops  -mfpu=neon -std=c++11
