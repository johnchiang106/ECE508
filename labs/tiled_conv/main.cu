#include "helper.hpp"

#define TILE_WIDTH 8
#define MAP_DIVISION 1
#define MASK_HEIGHT 5
#define MASK_WIDTH 5
#define CHANNEL 1
#define SM_IN_W (TILE_WIDTH + MASK_WIDTH - 1)
#define SM_IN_H (TILE_WIDTH + MASK_HEIGHT - 1)
#define MASK_SIZE 32*1*5*5 // MAP_SIZE*CHANNEL*MASK_HEIGHT*MASK_WIDTH
__constant__ float Const_Mask [MASK_SIZE];

// Sequential code for the forward path of the convolution layer
// You should not modify this code
static void conv_forward_valid(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {
  std::fill(Y, Y + ydims.flattened_length(), 0);

  for (auto i : range(0, ydims.num)) {
    for (auto m : range(0, ydims.depth )) {   // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width )) {
          const auto yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth )) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {   // filter height
              for (auto q : range(0, wdims.width )) { // filter width
                const auto xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const auto woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Baseline GPU kernel code for forward convolution.
// One thread per output index
// You should not modify this kernel as it is used for correctness comparison.
// Instead, define a new one below
__global__ void conv_forward_baseline_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {


  const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gx; i < ydims.num * ydims.depth * ydims.height * ydims.width; i += blockDim.x * gridDim.x) {
    Y[i] = 0.f;
  }

  for (size_t i = gx; i < ydims.num; i += gridDim.x * blockDim.x) {
    for (auto m : range(0, ydims.depth )) { // for each output feature map
      for (auto h : range(0, ydims.height)) { // for each output element
        for (auto w : range(0, ydims.width )) {
          const size_t yoffset = ((i * ydims.depth + m) * ydims.height + h) * ydims.width + w;
          for (auto c : range(0, xdims.depth )) {     // sum over all input feature maps
            for (auto p : range(0, wdims.height)) {   // filter height
              for (auto q : range(0, wdims.width )) { // filter width
                const size_t xoffset = ((((i * xdims.depth) + c) * xdims.height) + (h + p)) * xdims.width + (w + q);
                const size_t woffset = ((((m * wdims.depth) + c) * wdims.height) + p) * wdims.width + q;
                Y[yoffset] += X[xoffset] * W[woffset];
              }
            }
          }
        }
      }
    }
  }
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_baseline(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {

  dim3 dimGrid(1);
  dim3 dimBlock(32);

  conv_forward_baseline_kernel<<<dimGrid, dimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());

}

// Implement your optimized kernel here.
// Make any modifications you wish.
// Don't forget to modify the host code below, if needed!
__global__ void conv_forward_opt_kernel(const float *X, const shape xdims, const float *W, const shape wdims, float *Y, const shape ydims) {

  #define Channel xdims.depth
  #define Height xdims.height
  #define Width xdims.width
  #define mChannel wdims.depth
  #define mHeight wdims.height
  #define mWidth wdims.width
  #define Batch ydims.num
  #define Map_out ydims.depth
  #define Height_out ydims.height
  #define Width_out ydims.width

  __shared__ float SM_Input [CHANNEL*SM_IN_H*SM_IN_W];
  
  // const int BlockSize = MAP_DIVISION * TILE_WIDTH * TILE_WIDTH;
  // const int InputSize = Batch * Channel * Height * Width;
  // const int msize = mChannel * mHeight * mWidth;
  #define BlockSize MAP_DIVISION * TILE_WIDTH * TILE_WIDTH
  // #define BlockSize TILE_WIDTH * TILE_WIDTH
  #define InputSize Batch * Channel * Height * Width
  #define msize mChannel * mHeight * mWidth
  const int SM_Width = TILE_WIDTH + mWidth - 1;
  const int SM_ChSize = (TILE_WIDTH + mHeight - 1) * SM_Width;
  const int SM_InputSize = Channel * SM_ChSize;

  #define in_idx(b, c, h, w) ((b) * (Channel * Height * Width) + (c) * (Height * Width) + (h) * (Width) + w)
  #define mask_4d(m, c, h, w) Const_Mask[(m) * (msize) + (c) * (mHeight * mWidth) + (h) * (mWidth) + w]
  #define out_4d(b, m, h, w) Y[(b) * (Map_out * Height_out * Width_out) + (m) * (Height_out * Width_out) + (h) * (Width_out) + w]
  #define sm_in_idx(c, h, w) ((c)*(SM_ChSize) + (h)*(SM_Width) + w)

  int W_size = ceil(1.0 * Width_out / TILE_WIDTH);

  // bdx = ceil(1.0 * Map_out / MAP_DIVISION);
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;//, bdx = blockDim.x;
  // int m = blockIdx.x;
  int h_offset = (blockIdx.y / W_size) * TILE_WIDTH;
  int w_offset = (blockIdx.y % W_size) * TILE_WIDTH;
  // int m_offset = bdx * tz;
  int b = blockIdx.z;

  //@@ YOUR CODE HERE!
  for(int newIdx = tz*TILE_WIDTH*TILE_WIDTH + ty*TILE_WIDTH + tx; newIdx < SM_InputSize; newIdx += BlockSize){
    int c = newIdx / SM_ChSize;
    int newY = (newIdx % SM_ChSize) / SM_Width;
    int newX = newIdx % SM_Width;
    int index = in_idx(b,c,newY+h_offset,newX+w_offset);
    if(index < InputSize)
      SM_Input[sm_in_idx(c,newY,newX)] = X[index];
  }
  __syncthreads();
  if(h_offset + ty < Height_out && w_offset + tx < Width_out){
    // size_t mask_idx = 0;
    for (int m = tz; m < Map_out; m += MAP_DIVISION) {
    // for (int m = m_offset; m < Map_out && m < Map_out; m++) {
      // for each output feature map
      float acc = 0.0f;
      // size_t sm_idx = ty * SM_Width + tx;
      for (auto c : range(0, Channel)) {  // sum over all input feature maps
        for (auto p : range(0, mHeight)) {   // filter height
          for (auto q : range(0, mWidth)) { // filter width
            // acc += SM_Input[sm_idx++] * Const_Mask[mask_idx++];
            acc += SM_Input[sm_in_idx(c, ty+p, tx+q)] * mask_4d(m, c, p, q);
          }
          // sm_idx += SM_Width - mWidth;
        }
        // sm_idx += SM_ChSize - mHeight * SM_Width;
      }
      out_4d(b, m, h_offset + ty, w_offset + tx) = acc;
      // mask_idx += (MAP_DIVISION - 1) * msize;
    }
  }

  #undef out_4d
  #undef in_4d
  #undef mask_4d
  #undef Channel
  #undef Height
  #undef Width
  #undef mChannel
  #undef mHeight
  #undef mWidth
  #undef Map_out
  #undef Height_outs
  #undef Width_out
  #undef Batch
  #undef BlockSize
  #undef InputSize
}

// Host code to configure baseline GPU kernel
static void convlayer_gpu_opt(const float *X, const shape &xdims, const float *W, const shape &wdims, float *Y, const shape &ydims) {
  // int yd = ydims.depth, wd = wdims.depth, wh = wdims.height, ww = wdims.width, xd = xdims.depth;
  // printf("%d, %d, %d, %d, %d\n", yd, wd, wh, ww, xd);

  // ydims.depth	Map_out or output feature map
  // ydims.num	Batch
  // ydims.height	Height_out
  // ydims.width	Width_out
  // xdims.depth	Channel or input feature maps
  // xdims.height	Height
  // xdims.width	Width
  // wdims.height	K or filter height
  // wdims.width	K or filter width
  // X	device_input
  // Y	device_output
  // W	device_mask
  
  // int W_out = Width - K + 1, H_out = Height - K + 1;
  int W_size = ceil(1.0 * ydims.width / TILE_WIDTH);
  int H_size = ceil(1.0 * ydims.height / TILE_WIDTH);
  // int M_size = ceil(1.0 * ydims.depth / MAP_DIVISION);

  // dim3 DimGrid(ydims.depth,W_size*H_size,ydims.num);
  // dim3 DimGrid(M_size,W_size*H_size,ydims.num);
  dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,MAP_DIVISION);
  dim3 DimGrid(1,W_size*H_size,ydims.num);
  // dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1);
  // dim3 DimGrid(1);
  // dim3 DimBlock(32);

  // Modify this code to configure your optimized kernel.
  //@@ YOUR CODE HERE!!!
  conv_forward_opt_kernel<<<DimGrid, DimBlock>>>(X, xdims, W, wdims, Y, ydims);
  THROW_IF_ERROR(cudaGetLastError());

}


static int eval(const shape wDims, const shape xDims, bool doVerify) {

  // Generate model
  const auto conf_info = std::string("conv[wDims:") + std::to_string(wDims.num) + "," +
                                                      std::to_string(wDims.depth) + "," +
                                                      std::to_string(wDims.height) + "," +
                                                      std::to_string(wDims.width) +
                                                      " xDims:" + std::to_string(xDims.num) + "," +
                                                      std::to_string(xDims.depth) + "," +
                                                      std::to_string(xDims.height) + "," +
                                                      std::to_string(xDims.width) + "]";
  INFO("Running "  << conf_info);

  // Generate convolution weights
  float *hostW = allocate<float>(wDims);
  generate_convfilters(hostW, wDims);

  // generate input feature map
  float *hostX = allocate<float>(xDims);
  generate_data(hostX, xDims);

  // generate output feature map for verification
  const shape ydims = {xDims.num, wDims.num, (xDims.height - wDims.height + 1),
      (xDims.width - wDims.width + 1)};
  INFO("Allocating output tensor [" << ydims.num << "," << ydims.depth << "," << ydims.height << "," << ydims.width << "]");
  float *hostY = allocate<float>(ydims);
  float *expected = allocate<float>(ydims);
  generate_data(hostY, ydims);


  const size_t wByteCount = wDims.flattened_length() * sizeof(float);
  // int wsize = wDims.flattened_length();
  // printf("%d\n", wsize);
  const size_t xByteCount = xDims.flattened_length() * sizeof(float);
  const size_t yByteCount = ydims.flattened_length() * sizeof(float);

  float *deviceW = nullptr, *deviceX = nullptr, *deviceY = nullptr;
  timer_start("Allocating GPU memory.");
  THROW_IF_ERROR(cudaMalloc((void **)&deviceW, wByteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceX, xByteCount));
  THROW_IF_ERROR(cudaMalloc((void **)&deviceY, yByteCount));
  timer_stop();


  timer_start("Copying inputs to the GPU.");
  THROW_IF_ERROR(cudaMemcpy(deviceW, hostW, wByteCount, cudaMemcpyDefault));
  THROW_IF_ERROR(cudaMemcpyToSymbol(Const_Mask, hostW, wByteCount));
  THROW_IF_ERROR(cudaMemcpy(deviceX, hostX, xByteCount, cudaMemcpyDefault));
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  convlayer_gpu_opt(deviceX, xDims, deviceW, wDims, deviceY, ydims);
  THROW_IF_ERROR(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  if (doVerify) {
    timer_start("Copying output to the CPU");
    THROW_IF_ERROR(cudaMemcpy(hostY, deviceY, yByteCount, cudaMemcpyDefault));
    timer_stop();

    convlayer_gpu_baseline(deviceX, xDims, deviceW, wDims, deviceY, ydims);
    THROW_IF_ERROR(cudaDeviceSynchronize());
    THROW_IF_ERROR(cudaMemcpy(expected, deviceY, yByteCount, cudaMemcpyDefault));
    // conv_forward_valid(hostX, xDims, hostW, wDims, expected, ydims);
    verify(expected, hostY, ydims);
  }

  THROW_IF_ERROR(cudaFree(deviceW));
  THROW_IF_ERROR(cudaFree(deviceX));
  THROW_IF_ERROR(cudaFree(deviceY));
  free(hostW);
  free(hostX);
  free(hostY);
  free(expected);

  return 0;
}



TEST_CASE("Convlayer", "[convlayer]") {
#if 1
  // test five times in case code errors depend on data
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
  SECTION("[wDims:32,1,5,5 xDims:20,1,28,28]") {
    eval({32,1,5,5}, {20,1,28,28}, true);
  }
#else
  SECTION("[wDims:32,1,5,5 xDims:50000,1,28,28]") {
    eval({32,1,5,5}, {50000,1,28,28}, false);
  }
#endif
}
