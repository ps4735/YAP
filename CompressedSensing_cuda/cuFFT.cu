#include <cuda_runtime.h>
#include <cufft.h>
#include <cusp/complex.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <thrust\device_ptr.h>
#include <thrust\device_vector.h>

using namespace cusp;
using namespace thrust;

const unsigned int TILE_DIM = 32;
const unsigned int BLOCK_ROWS = 32;


__global__ void fftshift(complex<float> * d_idata, complex<float> * d_odata)
{

	__shared__ cusp::complex<float> tile[TILE_DIM][TILE_DIM + 1];
	int x = TILE_DIM * blockIdx.x + threadIdx.x;
	int y = TILE_DIM * blockIdx.y + threadIdx.y;
	int width = gridDim.x * TILE_DIM;
	tile[threadIdx.y][threadIdx.x] = d_idata[y*width + x];
	__syncthreads();

	if (blockIdx.x < width / (TILE_DIM * 2) && blockIdx.y < width / (TILE_DIM * 2))
	{
		d_odata[(y + width / 2)*width + x + width / 2] = tile[threadIdx.y][threadIdx.x];
	}
	if (blockIdx.x < width / (TILE_DIM * 2) && blockIdx.y >= width / (TILE_DIM * 2))
	{
		d_odata[(y - width / 2)*width + x + width / 2] = tile[threadIdx.y][threadIdx.x];
	}
	if (blockIdx.x >= width / (TILE_DIM * 2) && blockIdx.y < width / (TILE_DIM * 2))
	{
		d_odata[(y + width / 2)*width + x - width / 2] = tile[threadIdx.y][threadIdx.x];
	}
	if (blockIdx.x >= width / (TILE_DIM * 2) && blockIdx.y >= width / (TILE_DIM * 2))
	{
		d_odata[(y - width / 2)*width + x - width / 2] = tile[threadIdx.y][threadIdx.x];
	}
}




__global__  void ComplexScale(complex<float>* d_idata, complex<float>* d_odata, float scale, unsigned int width, unsigned int height)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col< width && row < height)
	{
		d_odata[row*width + col] = d_idata[row*width + col] * scale;
	}
}

extern "C" void Fft(complex<float>* d_raw, complex<float>* buffer, complex<float>* d_odata, bool fft_forward, cufftHandle& plan, unsigned int width, unsigned int height, unsigned int pitch)
{
	dim3 dimBlock(width, 1, 1);
	dim3 dimGrid(1, height, 1);
	dim3 grid(width / TILE_DIM, height / TILE_DIM, 1);
	dim3 block(TILE_DIM, BLOCK_ROWS, 1);
	fftshift << <grid, block >> >(d_raw, buffer);
	if (fft_forward)
	{
		checkCudaErrors(cufftExecC2C(plan, buffer, d_odata, CUFFT_FORWARD));
	}
	else
	{
		checkCudaErrors(cufftExecC2C(plan, buffer, d_odata, CUFFT_INVERSE));
	}

	fftshift << <grid, block >> >(d_odata, buffer);
	float scale = 1 / (float)std::sqrt(width * height);
	ComplexScale << <dimGrid, dimBlock >> >(buffer, d_odata, scale, width, height);
}
