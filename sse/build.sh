#!/bin/sh
gcc asm.c memcpy_bench.c -O3 -o memcpy -fopenmp
