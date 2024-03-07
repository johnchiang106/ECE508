#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

__device__ uint64_t linearCount(const uint32_t *const rowIdx, const uint32_t *const colIdx, const uint32_t *const rowPtr, uint32_t& uPtr, uint32_t& uEnd, uint32_t& vPtr, uint32_t& vEnd) {
  uint64_t tc = 0;
  uint32_t w1 = colIdx[uPtr];
  uint32_t w2 = colIdx[vPtr];

  while ((uPtr < uEnd) && (vPtr < vEnd)){
    if (w1 < w2){
      w1 = colIdx[++uPtr];
    }
    else if (w1 > w2){
      w2 = colIdx[++vPtr];
    }
    else if (w1 == w2){
      w1 = colIdx[++uPtr];
      w2 = colIdx[++vPtr];
      ++tc;
    }
  }
  return tc;
}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const rowIdx,         //!< node ids for edge srcs
                                 const uint32_t *const colIdx,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= numEdges)
    return;

  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  uint32_t uPtr = rowPtr[rowIdx[i]];
  uint32_t uEnd = rowPtr[rowIdx[i] + 1];
  uint32_t vPtr = rowPtr[colIdx[i]];
  uint32_t vEnd = rowPtr[colIdx[i] + 1];
  triangleCounts[i] = linearCount(rowIdx, colIdx, rowPtr, uPtr, uEnd, vPtr, vEnd);
}

__device__ uint64_t binarySearchCount(const uint32_t *const rowIdx, const uint32_t *const colIdx, const uint32_t *const rowPtr, uint32_t& uPtr, uint32_t& uEnd, uint32_t& vPtr, uint32_t& vEnd) {
  // if(vEnd - vPtr < uEnd - uPtr){
  //   return binarySearchCount(rowIdx, colIdx, rowPtr, vPtr, vEnd, uPtr, uEnd);
  // }
  uint64_t tc = 0;

  while ((uPtr < uEnd) && (vPtr < vEnd)){
    int l = vPtr, r = vEnd-1, mid, target = colIdx[uPtr++];
    while(l <= r){
      mid = l + (r-l)/2;
      if(colIdx[mid] > target)
        r = mid-1;
      else if(colIdx[mid] < target)
        l = mid+1;
      else{
        ++tc, ++mid;
        break;
      }
    }
    vPtr = (mid < l) ? l : mid;
  }

  return tc;
}

__global__ static void kernel_tc_rs(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const rowIdx,         //!< node ids for edge srcs
                                 const uint32_t *const colIdx,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Determine the source and destination node for the edge
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= numEdges)
    return;

  // Use the row pointer array to determine the start and end of the neighbor list in the column index array
  uint32_t uPtr = rowPtr[rowIdx[i]];
  uint32_t uEnd = rowPtr[rowIdx[i] + 1];
  uint32_t vPtr = rowPtr[colIdx[i]];
  uint32_t vEnd = rowPtr[colIdx[i] + 1];
  if(vEnd - vPtr < uEnd - uPtr){
    //swap if length of v is shorter
    uint32_t temp = uPtr;
    uPtr = vPtr, vPtr = temp;
    temp = uEnd, uEnd = vEnd, vEnd = temp;
  }
  //using binary search when V was as least 64 and V/U was at least 6
  // if(vEnd-vPtr >= 64 && (vEnd-vPtr)/(uEnd-uPtr) >= 6)
    triangleCounts[i] = binarySearchCount(rowIdx, colIdx, rowPtr, uPtr, uEnd, vPtr, vEnd);
  // else
  //   triangleCounts[i] = linearCount(rowIdx, colIdx, rowPtr, uPtr, uEnd, vPtr, vEnd);
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  int numEdges = view.nnz();
  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  pangolin::Vector<uint64_t> triangleCounts(numEdges, 0);
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.

  uint64_t total = 0;

  dim3 dimBlock(512);
  //@@ calculate the number of blocks needed
  dim3 dimGrid (ceil(1.0 * numEdges / dimBlock.x));

  if (mode == 1) {

    // num_rows() returns the number of rows
    // nnz() returns the number of non-zero elements (edges of the graph)
    // row_ptr() returns a pointer to the CSR row pointer array (length = num_rows() + 1 or 0 if num_rows() == 0).
    // col_ind() returns a pointer to the CSR column index array (length = nnz()).
    // row_ind() returns a pointer to the CSR row index array (length = nnz()).

    // uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
    // const uint32_t *const rowIdx,         //!< node ids for edge srcs
    // const uint32_t *const colIdx,         //!< node ids for edge dsts
    // const uint32_t *const rowPtr,      //!< source node offsets in edgeDst
    // const size_t numEdges
    //@@ launch the linear search kernel here
    kernel_tc<<<dimGrid, dimBlock>>>(triangleCounts.data(),view.row_ind(),view.col_ind(),view.row_ptr(),numEdges);
    cudaDeviceSynchronize();

  } else if (2 == mode) {

    //@@ launch the hybrid search kernel here
    kernel_tc_rs<<<dimGrid, dimBlock>>>(triangleCounts.data(),view.row_ind(),view.col_ind(),view.row_ptr(),numEdges);
    cudaDeviceSynchronize();

  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }

  //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
  for(int i = 0; i < numEdges; ++i){
    total += triangleCounts.data()[i];
  }
  return total;
}
