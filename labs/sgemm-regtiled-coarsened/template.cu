#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 64   //T
#define TILE_SZ_B 16    //U
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B) //S

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
    *
    * Compute C = A x B
    *   where A is a (m x k) matrix
    *   where B is a (k x n) matrix
    *   where C is a (m x n) matrix
    *
    * Use register and shared memory tiling and thread coarsening
    *
    * NOTE: A and C are column major, B is row major
    *
    ********************************************************************/

    // Macros for accessing flattened matrices
    #define A(row,col) A[(row) + (col)*m]
    #define B(row,col) B[(row)*n + (col)]
    #define C(row,col) C[(row) + (col)*m]

    __shared__ float SM[TILE_SZ_RATIO][TILE_SZ_B];
    // float output[TILE_SZ_B];
    float output[TILE_SZ_B] = {0};
    int tx = threadIdx.x;
    int Bx_offset = blockIdx.x * TILE_SZ_B;
    int Ay = blockIdx.y * TILE_SZ_A + tx;

    // INSERT KERNEL CODE HERE
    for(int Ax_offset = 0; Ax_offset < k; Ax_offset += TILE_SZ_RATIO){
        //read tile B
        int&& Bx = Bx_offset + tx % TILE_SZ_B;
        int&& By = Ax_offset + tx / TILE_SZ_B;
        if(Bx < n && By < k)
            SM[tx/TILE_SZ_B][tx%TILE_SZ_B] = B(By, Bx);
        __syncthreads();
        //loop to read value of M and compute
        if(Ay < m){
            for(int i = 0; i < TILE_SZ_RATIO && Ax_offset+i < k; ++i){
                float curr = A(Ay, Ax_offset+i);
                for(int j = 0; j < TILE_SZ_B; ++j){
                    output[j] += curr * SM[i][j];
                }
            }
        }
        __syncthreads();
    }

    //store output
    if(Ay >= m)
        return;
    for(int j = 0; j < TILE_SZ_B; ++j){
        C(Ay,Bx_offset+j) = output[j];
    }

    // SSL Hint (9/6/21) slides p36: try using just one register for the tile of A rather than several--in other words, load one value (per thread) from A and compute using that value rather than loading all values before doing the computation.  This approach seems to be slightly faster than the alternative.
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------
    // Your code need only consider the m, n, k, A, B, and C parameters of
    // the function, which provide the matrix sizes (m, n, k) and data
    // (A, B, C).

    //INSERT CODE HERE
    // #define TILE_SZ_A 128   //T
    // #define TILE_SZ_B 16    //U
    // #define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B) //S
    int Y_size = ceil(1.0 * m / TILE_SZ_A);
    int X_size = ceil(1.0 * n / TILE_SZ_B);
    dim3 DimGrid(X_size,Y_size,1);
    dim3 DimBlock(TILE_SZ_A,1,1);

    // Invoke CUDA kernel -----------------------------------------------------
    mysgemm<<<DimGrid,DimBlock>>>(m, n, k, A, B, C);

    //INSERT CODE HERE

}

