#include <fstream> // File I/O streams.
#include <iostream> // Input/Output streams.
#include <sstream> // String steam processing.
#include <stdio.h> // C-style I/O
#include <vector> // Dynamic array container.
#include <string> // String manipulation
#include <time.h> // Time handling.
#include <math.h> // Mathematical functions
#include <random> // C++11 random number generation
#include <iomanip> // I/O formatting
#include "cuda_runtime.h" // Main header for CUDA Runtime API.
#include <cuda.h> // Low-level CUDA Driver API (more control than Runtime API).
#include <curand_kernel.h> // Random number generation in CUDA kernels.




class Particle{
    real* x;
    real* y;
    real* fx;
    real* fy;
    real* x_UpdateHL;
    real* y_UpdateHL;
}