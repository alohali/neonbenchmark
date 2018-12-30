<<<<<<< HEAD
g++   -fopenmp -ofast -flax-vector-conversions  -funroll-loops   test_gemv.cpp  -o gemv    -std=c++11
=======
g++    -Ofast test_gemv.cpp  -o gemv -flax-vector-conversions  -funroll-loops  -mfpu=neon -std=c++11
>>>>>>> cc181bd8944238c73858b9869c246a2878c1132f
