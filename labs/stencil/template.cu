#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

#define TILE_SIZE 32

__global__ void kernel(int *A0, int *Anext, int Nx, int Ny, int Nz) {

  #define _in(i, j, k) A0[((k)*Ny + (j)) * Nx + (i)]
  #define _out(i, j, k) Anext[((k)*Ny + (j)) * Nx + (i)]

  // INSERT KERNEL CODE HERE
  __shared__ int ds_A[TILE_SIZE][TILE_SIZE];

  int tx = threadIdx.x, ty = threadIdx.y;
  int i = blockIdx.x * blockDim.x + tx;
  int j = blockIdx.y * blockDim.y + ty;
  int previous = _in(i, j, 0);
  int current = _in(i, j, 1);
  int next = _in(i, j, 2);
  for (int k = 1; k < Nz - 1; k++){
    // Copy current into shared memory.
    if (tx >= 0 && tx < Nx && ty >= 0 && ty < Ny)
      ds_A[tx][ty] = _in(i, j, k);
    // Barrier: wait until finished using shared memory.
    __syncthreads();
    if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1) {
      // _out(i,j,k) = -6 * current + previous + next + _in(i-1, j, k) + _in(i+1, j, k) + _in(i, j-1, k) + _in(i, j+1, k);
      _out(i,j,k) = -6 * current + previous + next + 
      ((0 < ty) ? ds_A[tx][ty-1] : _in(i, j-1, k)) + 
      ((TILE_SIZE - 1 < ty) ? ds_A[tx][ty+1] : _in(i, j+1, k)) + 
      ((0 < tx) ? ds_A[tx-1][ty] : _in(i-1, j, k)) + 
      ((TILE_SIZE - 1 > tx) ? ds_A[tx+1][ty] : _in(i+1, j, k));
      // Add conditional terms for other neighbors.
    }
    // Barrier: wait until finished using shared memory.
    __syncthreads();
    previous = current;
    current = next;
    if (k+2 < Nz)
      next = _in(i, j, k+2);
  }
}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {

  // INSERT CODE HERE
  int Y_size = ceil(1.0 * ny / TILE_SIZE);
  int X_size = ceil(1.0 * nx / TILE_SIZE);
  // dim3 DimGrid(16,16,1);
  dim3 DimGrid(X_size,Y_size,1);
  dim3 DimBlock(TILE_SIZE,TILE_SIZE,1);
  kernel<<<DimGrid,DimBlock>>>(A0,Anext,nx,ny,nz);
}


static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + 
                                                   std::to_string(ny) + "," + 
                                                   std::to_string(nz) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}



TEST_CASE("Stencil", "[stencil]") {

  SECTION("[dims:32,32,32]") {
    eval(32,32,32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30,30,30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29,29,29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31,31,31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29,29,29);
  }
  SECTION("[dims:1,1,2]") {
    eval(1,1,2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512,512,64);
  }

}
