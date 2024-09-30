#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include "../include/kernels_od.cuh"
#include "../include/kernels_od_intvar.cuh"

#include "../include/initialize_od.cuh"
#include "../include/gds.cuh"
#include "../include/utils.hpp"

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
/* This REALLY needs to be a library somewhere */
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* REFACTOR THIS ACCORDING TO REFACTORED LIBRARIES */
int main(int argc, char* argv[]){
	return 0;
}