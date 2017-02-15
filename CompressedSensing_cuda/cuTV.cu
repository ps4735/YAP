#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cusp/complex.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

using namespace cusp;
using namespace thrust;

const unsigned int TILE_SIZE = 16; //32（16）
const unsigned int BLOCK_ROWS_SIZE = 16;//8（4）

__global__ void tv_vertical(cusp::complex<float>* d_A, cusp::complex<float>* d_B, unsigned int pitch, unsigned int imageW, unsigned int imageH)  //d_B为结果矩阵
{
	__shared__ cusp::complex<float> tile1[TILE_SIZE][TILE_SIZE + 1];
	__shared__ cusp::complex<float> tile2[TILE_SIZE][TILE_SIZE + 1];
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	tile1[threadIdx.y][threadIdx.x] = d_A[y*imageW + x];
	if (y < imageH - 1)
	{
		tile2[threadIdx.y][threadIdx.x] = d_A[(y + 1)*imageW + x];
	}
	__syncthreads();

	if (y < imageH - 1)
	{
		d_B[y*imageW + x] = tile2[threadIdx.y][threadIdx.x] - tile1[threadIdx.y][threadIdx.x];
	}
	if (y == imageH - 1)
		d_B[y*imageW + x] = 0.0;
}



__global__ void tv_horizontal(cusp::complex<float>* d_A, cusp::complex<float>* d_B, unsigned int pitch, unsigned int imageW, unsigned int imageH)  //d_B为结果矩阵
{
	__shared__ cusp::complex<float> tile1[TILE_SIZE][TILE_SIZE + 1];
	__shared__ cusp::complex<float> tile2[TILE_SIZE][TILE_SIZE + 1];
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;

	tile1[threadIdx.y][threadIdx.x] = d_A[y*imageW + x];
	if (x < imageH - 1)
	{
		tile2[threadIdx.y][threadIdx.x] = d_A[y*imageW + x + 1];
	}
	__syncthreads();

	if (x < imageH - 1)
	{
		d_B[y*imageW + x] = tile2[threadIdx.y][threadIdx.x] - tile1[threadIdx.y][threadIdx.x];
	}
	if (x == imageH - 1)
	{
		d_B[y*imageW + imageH - 1] = 0.0;
	}
}

__global__ void itv(cusp::complex<float>* d_A, cusp::complex<float>* d_B, cusp::complex<float>* d_C, unsigned int imageW, unsigned int imageH, unsigned int pitch)   //d_C为结果矩阵
{
	__shared__ cusp::complex<float> tile1[TILE_SIZE][TILE_SIZE + 1];
	__shared__ cusp::complex<float> tile2[TILE_SIZE][TILE_SIZE + 1];
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;

	if (y == 0 || x == 0)
	{
		tile1[threadIdx.y][threadIdx.x] = 0.0;
		tile2[threadIdx.y][threadIdx.x] = 0.0;
	}

	if ((y > 0 && y < imageH - 1) && (x > 0 && x < imageW - 1))
	{
		tile1[threadIdx.y][threadIdx.x] = d_A[(y - 1)*imageW + x] - d_A[y*imageW + x];
		tile2[threadIdx.y][threadIdx.x] = d_B[y*imageW + x - 1] - d_B[y*imageW + x];
	}
	
	if ((y == imageH - 1) && (x == imageW - 1))
	{
		tile1[threadIdx.y][threadIdx.x] = d_A[(y - 1)*imageW + x];
		tile2[threadIdx.y][threadIdx.x] = d_B[y*imageW + x - 1];
	}

	__syncthreads();

	d_C[y*imageW + x] = tile1[threadIdx.y][threadIdx.x] + tile2[threadIdx.y][threadIdx.x];
}


extern "C" void FTV(complex<float>* d_A, device_vector<complex<float>>& d_B, device_vector<complex<float>>& d_C, unsigned int imageW, unsigned int imageH, unsigned int pitch)
{
	dim3 dimBlock(TILE_SIZE, BLOCK_ROWS_SIZE, 1);
	dim3 dimGrid(imageW / TILE_SIZE, imageH / TILE_SIZE, 1);
	tv_vertical << <dimGrid, dimBlock >> >(d_A, raw_pointer_cast(d_B.data()), pitch, imageW, imageH);
	tv_horizontal << <dimGrid, dimBlock >> >(d_A, raw_pointer_cast(d_C.data()), pitch, imageW, imageH);

}

extern "C" void IFTV(device_vector<complex<float>>& d_A, device_vector<complex<float>>& d_B, device_vector<complex<float>>& d_C, device_vector<complex<float>>& horizontal_buffer, device_vector<complex<float>>& vertical_buffer, unsigned int imageW, unsigned int imageH, unsigned int pitch) //d_C为结果矩阵
{
	dim3 dimBlock(TILE_SIZE, BLOCK_ROWS_SIZE, 1);
	dim3 dimGrid(imageW / TILE_SIZE, imageH / TILE_SIZE, 1);
	itv << <dimGrid, dimBlock >> >(raw_pointer_cast(d_A.data()), raw_pointer_cast(d_B.data()), raw_pointer_cast(d_C.data()), imageW, imageH, pitch);
}