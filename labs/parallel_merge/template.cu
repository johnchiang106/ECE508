//Circular did not work
#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/
__device__ int co_rank (int k, float* A, int A_len, float* B, int B_len) {
    int low = (k > B_len ? k - B_len : 0);
    int high = (k < A_len ? k : A_len);
    while (low < high) {
        int i = low + (high - low) / 2;
        int j = k - i;
        if (i > 0 && j < B_len && A[i - 1] > B[j]) {
            high = i - 1;
        } else if (j > 0 && i < A_len && A[i] <= B[j - 1]) {
            low = i + 1;
        } else {
            return i;
        }
    }
    return low;
}
/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int elt = ceil ((A_len+B_len)*1.0f/(BLOCK_SIZE*gridDim.x));
    int k_curr = tid * elt;
    if (A_len + B_len < k_curr) { k_curr = A_len + B_len; }
    int k_next = k_curr + elt;
    if (A_len + B_len < k_next) { k_next = A_len + B_len; }
    int i_curr = co_rank (k_curr, A, A_len, B, B_len);
    int i_next = co_rank (k_next, A, A_len, B, B_len);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential (&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    extern __shared__ float shareAB[];
    // __shared__ float shareAB[TILE_SIZE*2];
    __shared__ float shareC[TILE_SIZE];
    float* tileA = &shareAB[0];
    float* tileB = &shareAB[TILE_SIZE];
    int tx = threadIdx.x;
    int elt = ceil ((A_len + B_len) * 1.0f / gridDim.x);
    int blk_C_curr = blockIdx.x * elt;
    int blk_C_next = min(blk_C_curr + elt, A_len + B_len);
    if (tx == 0) {
        tileA[0] = co_rank (blk_C_curr, A, A_len, B, B_len);
        tileA[1] = co_rank (blk_C_next, A, A_len, B, B_len);
    }
    __syncthreads();
    int blk_A_curr = tileA[0], blk_A_next = tileA[1];
    int blk_B_curr = blk_C_curr - blk_A_curr;
    int blk_B_next = blk_C_next - blk_A_next;
    __syncthreads();
    int A_remaining = blk_A_next - blk_A_curr;
    int B_remaining = blk_B_next - blk_B_curr;
    int C_remaining = blk_C_next - blk_C_curr;
    while (C_remaining > 0) {
        // load tile
        for (int i = 0; i < TILE_SIZE; i += BLOCK_SIZE) {
            if (i + tx < A_remaining) {
                tileA[i + tx] = A[blk_A_curr + i + tx];
            }
            if (i + tx < B_remaining) {
                tileB[i + tx] = B[blk_B_curr + i + tx];
            }
        }
        __syncthreads();
        // process tile
        int per_thread = TILE_SIZE / BLOCK_SIZE;
        int thr_C_curr = min(tx * per_thread, C_remaining);
        int thr_C_next = min(thr_C_curr + per_thread, C_remaining);
        int A_in_tile = min(A_remaining, TILE_SIZE);
        int B_in_tile = min(B_remaining, TILE_SIZE);

        int thr_A_curr = co_rank(thr_C_curr, tileA, A_in_tile, tileB, B_in_tile);
        int thr_A_next = co_rank(thr_C_next, tileA, A_in_tile, tileB, B_in_tile);
        int thr_B_curr = thr_C_curr - thr_A_curr;
        int thr_B_next = thr_C_next - thr_A_next;
        merge_sequential(tileA + thr_A_curr, thr_A_next - thr_A_curr, tileB + thr_B_curr, thr_B_next - thr_B_curr, &shareC[thr_C_curr]);
        __syncthreads();
        // put tile C to global
        for (int i = 0; i < TILE_SIZE; i += BLOCK_SIZE) {
            if (i + tx < C_remaining) {
                C[blk_C_curr + i + tx] = shareC[i + tx];
            }
        }

        // advance variables for next tile
        if (tx == BLOCK_SIZE-1) {
            tileA[0] = thr_A_next;
            tileA[1] = thr_B_next;
        }
        __syncthreads();
        A_remaining -= tileA[0];
        B_remaining -= tileA[1];
        C_remaining -= TILE_SIZE;
        blk_A_curr += tileA[0];
        blk_B_curr += tileA[1];
        blk_C_curr += TILE_SIZE;
        __syncthreads();
    }
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__device__ int circular_co_rank (int k, float* A, int A_rear, int A_len, float* B, int B_rear, int B_len) {
    #define aIdx(i) ((A_rear + i) % A_len)
    #define bIdx(j) ((B_rear + j) % B_len)
    int low = (k > B_len ? k - B_len : 0);
    int high = (k < A_len ? k : A_len);
    while (low < high) {
        int i = low + (high - low) / 2;
        int j = k - i;
        if (i > 0 && j < B_len && A[aIdx(i-1)] > B[bIdx(j)]) {
            high = i - 1;
        } else if (j > 0 && i < A_len && A[aIdx(i)] <= B[bIdx(j-1)]) {
            low = i + 1;
        } else {
            return i;
        }
    }
    return low;
    #undef aIdx
    #undef bIdx
}
__device__ void circular_merge_sequential(float* A, int A_rear, int A_len, float* B, int B_rear, int B_len, float* C) {
    #define aIdx(i) ((A_rear + i) % A_len)
    #define bIdx(j) ((B_rear + j) % B_len)
    int i = 0, j = 0, k = 0;
    while ((i < A_len) && (j < B_len)) {
        if(A[aIdx(i)] <= B[bIdx(j)]){
            C[k++] = A[aIdx(i)];
            ++i;
        }
        else{
            C[k++] = B[bIdx(j)];
            ++j;
        }
        // C[k++] = A[aIdx(i)] <= B[bIdx(j)] ? A[aIdx(i)] : B[bIdx(j)];
        // // C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
        // ++i, ++j;
    }
    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[bIdx(j)];
            ++j;
        }
    } else {
        while (i < A_len) {
            C[k++] = A[aIdx(i)];
            ++i;
        }
    }
    #undef aIdx
    #undef bIdx
}

__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    extern __shared__ float shareAB[];
    __shared__ int shareA, shareB;
    __shared__ float shareC[TILE_SIZE];
    float* tileA = &shareAB[0];
    float* tileB = &shareAB[TILE_SIZE];
    int tx = threadIdx.x;
    int elt = ceil ((A_len + B_len) * 1.0f / gridDim.x);
    int blk_C_curr = blockIdx.x * elt;
    int blk_C_next = min(blk_C_curr + elt, A_len + B_len);
    if (tx == 0) {
        shareA = co_rank (blk_C_curr, A, A_len, B, B_len);
        shareB = co_rank (blk_C_next, A, A_len, B, B_len);
    }
    __syncthreads();
    int blk_A_curr = shareA, blk_A_next = shareB;
    int blk_B_curr = blk_C_curr - blk_A_curr;
    int blk_B_next = blk_C_next - blk_A_next;
    __syncthreads();
    int A_remaining = blk_A_next - blk_A_curr;
    int B_remaining = blk_B_next - blk_B_curr;
    int C_remaining = blk_C_next - blk_C_curr;
    int rearA = 5, rearB = 5;
    while (C_remaining > 0) {
        // load tile
        for (int i = 0; i < TILE_SIZE; i += BLOCK_SIZE) {
            if (i + tx < A_remaining) {
                tileA[(rearA+i+tx)%TILE_SIZE] = A[blk_A_curr + i + tx];
            }
            if (i + tx < B_remaining) {
                tileB[(rearB+i+tx)%TILE_SIZE] = B[blk_B_curr + i + tx];
            }
        }
        __syncthreads();
        // process tile
        int per_thread = TILE_SIZE / BLOCK_SIZE;
        int thr_C_curr = min(tx * per_thread, C_remaining);
        int thr_C_next = min(thr_C_curr + per_thread, C_remaining);
        int A_in_tile = min(A_remaining, TILE_SIZE);
        int B_in_tile = min(B_remaining, TILE_SIZE);

        int thr_A_curr = circular_co_rank(thr_C_curr, tileA, rearA, A_in_tile, tileB, rearB, B_in_tile);
        int thr_A_next = circular_co_rank(thr_C_next, tileA, rearA, A_in_tile, tileB, rearB, B_in_tile);
        int thr_B_curr = thr_C_curr - thr_A_curr;
        int thr_B_next = thr_C_next - thr_A_next;
        circular_merge_sequential(tileA + thr_A_curr, rearA, thr_A_next - thr_A_curr, tileB + thr_B_curr, rearB, thr_B_next - thr_B_curr, &shareC[thr_C_curr]);
        __syncthreads();
        // put tile C to global
        for (int i = 0; i < TILE_SIZE; i += BLOCK_SIZE) {
            if (i + tx < C_remaining) {
                C[blk_C_curr + i + tx] = shareC[i + tx];
            }
        }

        // advance variables for next tile
        if (tx == BLOCK_SIZE-1) {
            shareA = thr_A_next;
            shareB = thr_B_next;
        }
        __syncthreads();
        A_remaining -= shareA;
        B_remaining -= shareB;
        C_remaining -= TILE_SIZE;
        blk_A_curr += shareA;
        blk_B_curr += shareB;
        blk_C_curr += TILE_SIZE;
        // rearA = (rearA + shareA) % TILE_SIZE;
        // rearB = (rearB + shareB) % TILE_SIZE;
        __syncthreads();
    }
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE, TILE_SIZE*2*sizeof(float)>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE, TILE_SIZE*2*sizeof(float)>>>(A, A_len, B, B_len, C);
}
