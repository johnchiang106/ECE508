#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512

#define MAX_CAPACITY 4096

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048 //4096

// Number of warp queues per block
#define NUM_WARP_QUEUES 8
// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY (BQ_CAPACITY / NUM_WARP_QUEUES)
// Number of threads works on the same warp queue
#define NUM_WARP_THREAD (BLOCK_SIZE / NUM_WARP_QUEUES)

__device__ bool storeNodes(unsigned int* targetQueue, unsigned int capacity, unsigned int* currNumNodes, const unsigned int& neighborID){
  if(*currNumNodes >= capacity){
    return false;
  }
  unsigned int num = atomicAdd(currNumNodes,1);
  if(num < capacity){
    targetQueue[num] = neighborID;
    return true;
  }
  return false;
}

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queueing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Loop over all nodes in the current level
  if(i < *numCurrLevelNodes){
    unsigned int node = currLevelNodes[i];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node+1]; ++nbrIdx) {
      unsigned int neighborID = nodeNeighbors[nbrIdx];
      // If neighbor hasn't been visited yet
      // Add neighbor to global queue
      if(nodeVisited[neighborID] >= 1 || 
          atomicExch(&nodeVisited[neighborID],1) >= 1)
        continue;
      unsigned int nodeNum = atomicAdd(numNextLevelNodes,1);
      nextLevelNodes[nodeNum] = neighborID;
    }
  }
}

__global__ void gpu_block_queueing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE
  unsigned int tx = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + tx;
  // Initialize shared memory queue (size should be BQ_CAPACITY)
  __shared__ unsigned int numQueueNodes;
  __shared__ unsigned int start;
  __shared__ unsigned int SM_queue[BQ_CAPACITY];
  if(tx == 0)  numQueueNodes = 0;
  __syncthreads();
  // Loop over all nodes in the current level
  if(i < *numCurrLevelNodes){
    unsigned int node = currLevelNodes[i];
    // unsigned int nodeNum;
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node+1]; ++nbrIdx) {
      unsigned int neighborID = nodeNeighbors[nbrIdx];
      // If neighbor hasn't been visited yet
      // Add neighbor to block queue
      if(nodeVisited[neighborID] >= 1 || 
          atomicExch(&nodeVisited[neighborID],1) >= 1)
        continue;
      // If full, add neighbor to global queue
      if(!storeNodes(SM_queue, BQ_CAPACITY, &numQueueNodes, neighborID)){
        storeNodes(nextLevelNodes, MAX_CAPACITY, numNextLevelNodes, neighborID);
      }
      /*
      if(numQueueNodes >= BQ_CAPACITY){
        nodeNum = atomicAdd(numNextLevelNodes,1);
        nextLevelNodes[nodeNum] = neighborID;
        continue;
      }
      nodeNum = atomicAdd(&numQueueNodes,1);
      if(nodeNum >= BQ_CAPACITY){
        nodeNum = atomicAdd(numNextLevelNodes,1);
        nextLevelNodes[nodeNum] = neighborID;
        continue;
      }
      SM_queue[nodeNum] = neighborID;
      //*/
    }
  }
  __syncthreads();

  // Allocate space for block queue to go into global queue
  if(tx == 0){
    numQueueNodes = min(numQueueNodes,BQ_CAPACITY);
    start = atomicAdd(numNextLevelNodes,numQueueNodes);
  }
  __syncthreads();
  // Store block queue in global queue
  for(unsigned int idx = tx; idx < numQueueNodes; idx += BLOCK_SIZE){
    nextLevelNodes[start+idx] = SM_queue[idx];
  }
}

__global__ void gpu_warp_queueing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // This version uses NUM_WARP_QUEUES warp queues of capacity 
  // WQ_CAPACITY.  Be sure to interleave them as discussed in lecture.  
  unsigned int tx = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + tx; 

  // Don't forget that you also need a block queue of capacity BQ_CAPACITY.
  // Initialize shared memory queues (warp and block)
  __shared__ unsigned int numQueueNodes;
  __shared__ unsigned int SM_queue[BQ_CAPACITY];
  __shared__ unsigned int start;
  __shared__ unsigned int numWarpNodes[NUM_WARP_QUEUES];
  __shared__ unsigned int warp_queue[WQ_CAPACITY][NUM_WARP_QUEUES];
  __shared__ unsigned int warp_start[NUM_WARP_QUEUES];

  const unsigned int warpQueueIdx = tx % NUM_WARP_QUEUES;
  const unsigned int warpIdx = tx / NUM_WARP_QUEUES;
  unsigned int nodeNum;
  if(tx == 0)  numQueueNodes = 0;
  if(warpIdx == 0)
    numWarpNodes[warpQueueIdx] = 0;
  __syncthreads();
  // Loop over all nodes in the current level
  if(i < *numCurrLevelNodes){
    unsigned int node = currLevelNodes[i];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node+1]; ++nbrIdx) {
      unsigned int neighborID = nodeNeighbors[nbrIdx];
      // If neighbor hasn't been visited yet
      // Add neighbor to the queue
      if(nodeVisited[neighborID] >= 1 || 
          atomicExch(&nodeVisited[neighborID],1) >= 1)
        continue;
      nodeNum = atomicAdd(&numWarpNodes[warpQueueIdx],1);
      if(nodeNum < WQ_CAPACITY){
        warp_queue[nodeNum][warpQueueIdx] = neighborID;
        continue;
      }
      // If full, add neighbor to block queue
      nodeNum = atomicAdd(&numQueueNodes,1);
      if(nodeNum < BQ_CAPACITY){
        SM_queue[nodeNum] = neighborID;
        continue;
      }
      // If full, add neighbor to global queue
      nodeNum = atomicAdd(numNextLevelNodes,1);
      nextLevelNodes[nodeNum] = neighborID;
    }
  }
  __syncthreads();

  // Allocate space for warp queue to go into block queue
  if(warpIdx == 0){
    numWarpNodes[warpQueueIdx] = min(numWarpNodes[warpQueueIdx],WQ_CAPACITY);
    warp_start[warpQueueIdx] = atomicAdd(&numQueueNodes,numWarpNodes[warpQueueIdx]);
    if(warp_start[warpQueueIdx] >= BQ_CAPACITY){
      warp_start[warpQueueIdx] = atomicAdd(numNextLevelNodes,numWarpNodes[warpQueueIdx]) + BQ_CAPACITY;
    }
  }
  __syncthreads();

  // Store warp queues in block queue (use one warp or one thread per queue)
  // Add any nodes that don't fit (remember, space was allocated above)
  //    to the global queue
  if(warp_start[warpQueueIdx] < BQ_CAPACITY){
    for(unsigned int idx = warpIdx; idx < numWarpNodes[warpQueueIdx]; idx += NUM_WARP_THREAD){
      nodeNum = warp_start[warpQueueIdx] + idx;
      if(nodeNum < BQ_CAPACITY){
        SM_queue[nodeNum] = warp_queue[idx][warpQueueIdx];
        continue;
      }
      // Saturate block queue counter (too large if warp queues overflowed)
      nodeNum = atomicAdd(numNextLevelNodes,1);
      nextLevelNodes[nodeNum] = warp_queue[idx][warpQueueIdx];
    }
  }
  else{
    for(unsigned int idx = warpIdx; idx < numWarpNodes[warpQueueIdx]; idx += NUM_WARP_THREAD){
      nodeNum = warp_start[warpQueueIdx] - BQ_CAPACITY + idx;
      nextLevelNodes[nodeNum] = warp_queue[idx][warpQueueIdx];
    }
  }
  __syncthreads();


  // Allocate space for block queue to go into global queue
  if(tx == 0){
    numQueueNodes = min(numQueueNodes,BQ_CAPACITY);
    start = atomicAdd(numNextLevelNodes,numQueueNodes);
  }
  __syncthreads();
  // Store block queue in global queue
  for(unsigned int idx = tx; idx < numQueueNodes; idx += BLOCK_SIZE){
    nextLevelNodes[start+idx] = SM_queue[idx];
  }
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queueing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queueing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
