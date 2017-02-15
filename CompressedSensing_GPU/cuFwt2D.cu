#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cusp/complex.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>


using namespace cusp;
using namespace thrust;

const unsigned int TILE_DIM = 32; //16（32）
const unsigned int BLOCK_ROWS = 32;//4（8）
const int KERNEL_LENGTH = 4;

__global__ void complex_copy(cusp::complex<float>* d_A, const cusp::complex<float>* d_B, unsigned int width, unsigned int height)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (row < height && col < width)
	{
		d_A[row*width + col] = d_B[row*width + col];
	}

}

__global__ void DownsampleLow(unsigned int filter_size, cusp::complex<float> *d_idata, cusp::complex<float> * d_odata, float * l_Kernel, unsigned int length, unsigned int width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	unsigned int width = pitch / sizeof(cusp::complex<float>);

	//Aconv
	extern __shared__ cusp::complex<float> s_dataL[];
	if (threadIdx.x < length)
	{
		s_dataL[x] = d_idata[y*width + x];
	}
	if (threadIdx.x >= length)
	{
		s_dataL[x] = d_idata[y*width + x - length];
	}
	__syncthreads();

	if (x < length && x % 2 == 0)
	{
		cusp::complex<float> sumL(0.0, 0.0);
		for (unsigned int i = 0; i < KERNEL_LENGTH; ++i)
		{
			sumL += s_dataL[threadIdx.x + i] * l_Kernel[i];                                     //d_odata size = padding size
		}

		d_odata[width*x / 2 + y] = sumL;
	}
}

__global__ void DownsampleHigh(unsigned int filter_size, cusp::complex<float> *d_idata, cusp::complex<float> * d_odata, float * h_Kernel, unsigned int length, unsigned int width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	unsigned int width = pitch / sizeof(cusp::complex<float>);
	//Iconv
	extern __shared__ cusp::complex<float> s_dataH[];
	s_dataH[x] = d_idata[y*width];
	if (threadIdx.x < filter_size - 1)
	{
		s_dataH[x] = d_idata[y*width + length - filter_size + 1 + x];
	}
	if (threadIdx.x >= filter_size && threadIdx.x < length + filter_size - 1)
	{
		s_dataH[x] = d_idata[y*width + x - filter_size + 1];
	}
	__syncthreads();

	if (threadIdx.x >= filter_size && threadIdx.x % 2 == 0)
	{
		cusp::complex<float> sumH(0.0, 0.0);
		for (unsigned int i = 0; i < KERNEL_LENGTH; ++i)
		{
			sumH += s_dataH[threadIdx.x - i] * h_Kernel[i];
		}
		d_odata[width*(x - filter_size + length) / 2 + y] = sumH;
	}
}


__global__ void IdwtDb1D(unsigned int filter_size, cusp::complex<float> *d_idata, cusp::complex<float> *d_odata, float * l_Kernel, float * h_Kernel, unsigned int length, unsigned int width)			 //d_dataH(参数d_odataH)shift之后变成shift_data,然后d_dataH作为输出的d_odataH
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	unsigned int width = pitch / sizeof(cusp::complex<float>);
	//ACONV(高通滤波)
	extern __shared__ cusp::complex<float> s_dataH[];
	s_dataH[y] = 0.0;
	if (y < length && y % 2 != 0)
	{
		s_dataH[y] = d_idata[((y - 1) / 2 + length / 2) * width + x];
	}
	if (y >= length & y % 2 != 0)
	{
		s_dataH[y] = d_idata[width*(y - 1) / 2 + x];
	}
	__syncthreads();

	if (y < length)
	{
		register cusp::complex<float> sumH(0.0, 0.0);
		for (unsigned int i = 0; i < KERNEL_LENGTH; ++i)
		{
			sumH += s_dataH[y + i] * h_Kernel[i];                    //d_odata size = padding size
		}
		d_odata[x * width + y] = sumH;
	}

	//ICONV(低通滤波)
	extern __shared__ cusp::complex<float> s_dataL[];
	s_dataL[y] = 0.0;
	if (y < filter_size && y % 2 == 0)
		s_dataL[y] = d_idata[(y + length - filter_size) / 2 * width + x];
	if (y >= filter_size && y % 2 == 0)
	{
		s_dataL[y] = d_idata[(y - filter_size) / 2 * width + x];
	}
	__syncthreads();
	if (y >= filter_size)
	{
		register cusp::complex<float> sumL(0.0, 0.0);
		for (unsigned int i = 0; i < KERNEL_LENGTH; ++i)
		{
			sumL += s_dataL[y - i] * l_Kernel[i];                    //d_odata size = padding size
		}
		d_odata[x * width + y - filter_size] += sumL;
	}
}

extern "C" void Fwt2D(complex<float> * d_raw, complex<float> * d_odata, unsigned int p, unsigned int scale, unsigned int J, unsigned int imageW, unsigned int imageH, float * l_Kernel, float * h_Kernel)
{
	const unsigned int nc = imageW;
	unsigned int mem_shared;
	dim3 block;
	dim3 grid;
	block.x = nc;			                     //初始值为256
	grid.y = nc;
	unsigned int length = nc;

	dim3 dimBlock(imageW, 1, 1);
	dim3 dimGrid(1, imageH, 1);
	complex_copy << <dimGrid, dimBlock>> >(d_odata, d_raw, imageW, imageH);

	device_vector<complex<float>>  d_tranp(nc*nc);
	for (unsigned int jscale = J - 1; jscale >= scale; --jscale)
	{
		block.x = length + p;
		block.y = 1;
		grid.x = 1;
		grid.y = length;
		mem_shared = (length + p)*sizeof(float) * 2;
		//行方向小波变换		
		DownsampleLow << <grid, block, mem_shared>> >(p, d_odata, raw_pointer_cast(d_tranp.data()), l_Kernel, length, imageW);
		DownsampleHigh << <grid, block, mem_shared>> >(p, d_odata, raw_pointer_cast(d_tranp.data()), h_Kernel, length, imageW);
		//列方向小波变换
		DownsampleLow << <grid, block, mem_shared>> >(p, raw_pointer_cast(d_tranp.data()), d_odata, l_Kernel, length, imageW);
		DownsampleHigh << <grid, block, mem_shared>> >(p, raw_pointer_cast(d_tranp.data()), d_odata, h_Kernel, length, imageW);
		length = length >> 1;
	}
}

extern "C" void IFwt2D(complex<float> * d_raw, complex<float> * d_odata, unsigned int p, unsigned int scale, unsigned int J, unsigned int imageW, unsigned int imageH, unsigned int np, float * l_Kernel, float * h_Kernel)
{
	unsigned int nc = imageW;
	unsigned int mem_shared;
	dim3 block;
	dim3 grid;
	unsigned int length = np;
	dim3 dimBlock(imageW, 1, 1);
	dim3 dimGrid(1, imageH, 1);
	complex_copy << <dimGrid, dimBlock>> >(d_odata, d_raw, imageW, imageH);

	device_vector<complex<float>> d_tranp(nc*nc);
	for (unsigned int jscale = scale; jscale <= J - 1; ++jscale)
	{

		mem_shared = (length + p)*sizeof(complex<float>) * 2;
		block.x = 1;
		block.y = length + p;
		grid.x = length;
		grid.y = 1;
		//上采样,行方向小波变换
		IdwtDb1D << <grid, block, mem_shared>> >(p, d_odata, raw_pointer_cast(d_tranp.data()), l_Kernel, h_Kernel, length, imageW);
		//上采样，列方向小波变换
		IdwtDb1D << <grid, block, mem_shared>> >(p, raw_pointer_cast(d_tranp.data()), d_odata, l_Kernel, h_Kernel, length, imageW);
		length = length << 1;
	}
}