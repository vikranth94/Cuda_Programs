/*
 * multShare.c
 *
 * Robert Hochberg
 * January 24, 2012
 *
 * Based nearly entirely on the code from the CUDA C Programming Guide
 */

#include "multShare.h"

static __inline__ uint64_t gettime(void) { 
  struct timeval tv; 
  gettimeofday(&tv, NULL); 
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec)); 
} 

static uint64_t usec;

__attribute__ ((noinline))  void begin_roi() {
  usec=gettime();
}

__attribute__ ((noinline))  void end_roi()   {
  usec=(gettime()-usec);
  std::cout << "elapsed (sec): " << usec/1000000.0 << "\n";
}

void compare(Matrix C, Matrix C_shared, int size) {
  bool error = false;
  for(int i = 0; i < size; ++i) {
      float diff = C.elements[i] - C_shared.elements[i];
      if(diff>0.001f || diff <-0.001f) {
      error = true; 
      break;
    }
  }
  if(error) {
    for(int i = 0; i < size; ++i) {
      std::cout << i << " " << C.elements[i] << ":" << C_shared.elements[i];;

      float diff = C.elements[i] - C_shared.elements[i];
      if(diff>0.001f || diff <-0.001f) {
        std::cout << " \t\tERROR";
      }
      std::cout << "\n";
    }
  } else {
    std::cout << "results match\n";
  }
}

// Matrix multiplication - Host code 
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE 
Matrix MatMul_Shared(const Matrix A, const Matrix B, Matrix C_shared) { 

  // Load A and B to device memory 
  Matrix d_A; 
  d_A.width = d_A.stride = A.width; 
  d_A.height = A.height; 
  size_t size = A.width * A.height * sizeof(float); 
  cudaError_t err = cudaMalloc(&d_A.elements, size); 
  printf("CUDA malloc A: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
  printf("Copy A to device: %s\n",cudaGetErrorString(err)); 

  Matrix d_B; 
  d_B.width = d_B.stride = B.width; 
  d_B.height = B.height; 
  size = B.width * B.height * sizeof(float); 
  err = cudaMalloc(&d_B.elements, size); 
  printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
  err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
  printf("Copy B to device: %s\n",cudaGetErrorString(err)); 

  // Allocate C in device memory 
  Matrix d_C; 
  d_C.width = d_C.stride = C_shared.width; 
  d_C.height = C_shared.height; 
  size = C_shared.width * C_shared.height * sizeof(float); 
  err = cudaMalloc(&d_C.elements, size); 
  printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

  // Invoke kernel 
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y); 
    begin_roi();
    MatMulSharedKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
    end_roi();
    err = cudaThreadSynchronize();
    printf("Run kernel: %s\n", cudaGetErrorString(err));

  // Read C from device memory 
  err = cudaMemcpy(C_shared.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
  printf("Copy C off of device: %s\n",cudaGetErrorString(err));

  // Free device memory
  cudaFree(d_A.elements); 
  cudaFree(d_B.elements); 
  cudaFree(d_C.elements); 
  return C_shared;
} 

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) { 
  return A.elements[row * A.stride + col]; 
} 

// Set a matrix element 
__device__ void SetElement(Matrix A, int row, int col, float value) { 
  A.elements[row * A.stride + col] = value; 
} 

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is 
// located col sub-matrices to the right and row sub-matrices down 
// from the upper-left corner of A 
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) { 
  Matrix Asub; 
  Asub.width = BLOCK_SIZE; 
  Asub.height = BLOCK_SIZE; 
  Asub.stride = A.stride; 
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col]; 
  return Asub; 
}


// Matrix multiplication kernel called by MatMul() 
__global__ void MatMulSharedKernel(Matrix A, Matrix B, Matrix C) { 
  // Block row and column 
  int blockRow = blockIdx.y; 
  int blockCol = blockIdx.x; 

  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol); 

  // Each thread computes one element of Csub 
  // by accumulating results into Cvalue 
  float Cvalue = 0.0; 

  // Thread row and column within Csub 
  int row = threadIdx.y; 
  int col = threadIdx.x; 

  // Loop over all the sub-matrices of A and B that are 
  // required to compute Csub 
  // Multiply each pair of sub-matrices together 
  // and accumulate the results 
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
    // Get sub-matrix Asub of A 
    Matrix Asub = GetSubMatrix(A, blockRow, m); 

    // Get sub-matrix Bsub of B 
    Matrix Bsub = GetSubMatrix(B, m, blockCol); 

    // Shared memory used to store Asub and Bsub respectively 
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; 
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE]; 

    // Load Asub and Bsub from device memory to shared memory 
    // Each thread loads one element of each sub-matrix 
    As[row][col] = GetElement(Asub, row, col); 
    Bs[row][col] = GetElement(Bsub, row, col); 

    // Synchronize to make sure the sub-matrices are loaded 
    // before starting the computation 
    __syncthreads(); 

    // Multiply Asub and Bsub together 
    for (int e = 0; e < BLOCK_SIZE; ++e) 
      Cvalue += As[row][e] * Bs[e][col];
 
    // Synchronize to make sure that the preceding 
    // computation is done before loading two new 
    // sub-matrices of A and B in the next iteration 
    __syncthreads();  
  }

  // Write Csub to device memory 
  // Each thread writes one element 
  Cvalue = (Cvalue>0) ? Cvalue : Cvalue/4;
  SetElement(Csub, row, col, Cvalue); 
}

void MatMul(const Matrix A, const Matrix B, Matrix C) { 

  // Load A and B to device memory 
  Matrix d_A; 
  d_A.width = A.width; 
  d_A.height = A.height; 
  size_t size = A.width * A.height * sizeof(float); 
  cudaError_t err = cudaMalloc(&d_A.elements, size); 
  printf("CUDA malloc A: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
  printf("Copy A to device: %s\n",cudaGetErrorString(err)); 
  
  Matrix d_B; 
  d_B.width = B.width; 
  d_B.height = B.height; 
  size = B.width * B.height * sizeof(float); 
  err = cudaMalloc(&d_B.elements, size); 
  printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
  err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
  printf("Copy B to device: %s\n",cudaGetErrorString(err)); 

  // Allocate C in device memory 
  Matrix d_C; 
  d_C.width = C.width; 
  d_C.height = C.height; 
  size = C.width * C.height * sizeof(float); 
  err = cudaMalloc(&d_C.elements, size); 
  printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

  // Invoke kernel 
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, 
         (A.height + dimBlock.y - 1) / dimBlock.y);
  begin_roi(); 
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
  end_roi();
  err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));
  
  // Read C from device memory 
  err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
  printf("Copy C off of device: %s\n",cudaGetErrorString(err));

  // Free device memory 
  cudaFree(d_A.elements); 
  cudaFree(d_B.elements); 
  cudaFree(d_C.elements); 
} 

// Matrix multiplication kernel called by MatMul() 
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) { 
  // Each thread computes one element of C 
  // by accumulating results into Cvalue 
  float Cvalue = 0.0; 
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x; 
  if(row > A.height || col > B.width) return;
  for (int e = 0; e < A.width; ++e) 
    Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]); 
  Cvalue = (Cvalue>0) ? Cvalue : Cvalue/4;
  C.elements[row * C.width + col] = Cvalue; 
}




int main(int argc, char* argv[]){
  Matrix A, B, C, C_shared;
  int a1, a2, b1, b2;
  a1 = atoi(argv[1]);			/* Height of A */
  a2 = atoi(argv[2]);			/* Width  of A */
  b1 = a2;		         	/* Height of B */
  b2 = atoi(argv[3]);			/* Width  of B */

  A.height = a1;
  A.width = a2;
  A.elements = (float*)malloc(A.width * A.height * sizeof(float));

  B.height = b1;
  B.width = b2;
  B.elements = (float*)malloc(B.width * B.height * sizeof(float));

  C.height = A.height;
  C.width = B.width;
  C.elements = (float*)malloc(C.width * C.height * sizeof(float));

  C_shared.height = A.height;
  C_shared.width = B.width;
  C_shared.elements = (float*)malloc(C.width * C.height * sizeof(float));

  for(int i = 0; i < A.height; i++)
    for(int j = 0; j < A.width; j++)
      A.elements[i*A.width + j] = (rand() % 3);

  for(int i = 0; i < B.height; i++)
    for(int j = 0; j < B.width; j++)
      B.elements[i*B.width + j] = (rand() % 2);
  C_shared = MatMul_Shared(A, B, C_shared);
  MatMul(A, B, C);
  compare(C, C_shared, b2);
  /*
  for(int i = 0; i < min(10, A.height); i++){
    for(int j = 0; j < min(10, A.width); j++)
      printf("%f ", A.elements[i*A.width + j]);
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < min(10, B.height); i++){
    for(int j = 0; j < min(10, B.width); j++)
      printf("%f ", B.elements[i*B.width + j]);
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < min(10, C.height); i++){
    for(int j = 0; j < min(10, C.width); j++)
      printf("%f ", C.elements[i*C.width + j]);
    printf("\n");
  }
  printf("\n");
  */
}
