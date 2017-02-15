#ifndef _CUDAMATRIXOP_H_
#define _CUDAMATRIXOP_H_
#include <cusp\complex.h>


const unsigned int TILE_DIM = 32; //16（32）
const unsigned int BLOCK_ROWS = 32;//4（8）
const int KERNEL_LENGTH = 4;
//convolution kernel storage
__constant__ float l_Kernel[KERNEL_LENGTH]; // 所谓的Db小波基低通采样
__constant__ float h_Kernel[KERNEL_LENGTH]; // 所谓的Db小波基高通采样

__global__ void DownsampleLow(unsigned int filter_size, cusp::complex<float> *d_idata, cusp::complex<float> * d_odata, unsigned int length, unsigned int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int width = pitch / sizeof(cusp::complex<float>);

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

__global__ void DownsampleHigh(unsigned int filter_size, cusp::complex<float> *d_idata, cusp::complex<float> * d_odata, unsigned int length, unsigned int pitch)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int width = pitch / sizeof(cusp::complex<float>);
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


__global__ void IdwtDb1D(unsigned int filter_size, cusp::complex<float> *d_idata, cusp::complex<float> *d_odata, unsigned int length, unsigned int pitch)			 //d_dataH(参数d_odataH)shift之后变成shift_data,然后d_dataH作为输出的d_odataH
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int width = pitch / sizeof(cusp::complex<float>);
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

///////////////////////////////重建算法中的运算操作定义/////////////////////////////////////////
__global__ void CalculateFidelity(cusp::complex<float>* d_A, cusp::complex<float>* d_B, float step, cusp::complex<float>* d_C, float *d_out, unsigned int lpitch, unsigned int rpitch, unsigned int imageW, unsigned int imageH)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	if (col< imageH && row < imageW)
	{

		d_out[row*rpitch / sizeof(float) + col] = cusp::norm(d_A[row*lpitch / sizeof(cusp::complex<float>) + col] * step + d_B[row*lpitch / sizeof(cusp::complex<float>) + col] - d_C[row*lpitch / sizeof(cusp::complex<float>) + col]);
	}
}

__global__ void norm_kernel(cusp::complex<float>* d_A, cusp::complex<float>* d_B, float* d_C, float* d_D, unsigned int lpitch, unsigned int rpitch, unsigned int imageW, unsigned int imageH)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < imageW && row < imageH)
	{
		d_C[row*rpitch / sizeof(cusp::complex<float>) + col] = cusp::norm(d_A[row*lpitch / sizeof(cusp::complex<float>) + col]);
		d_D[row*rpitch / sizeof(cusp::complex<float>) + col] = cusp::norm(d_B[row*lpitch / sizeof(cusp::complex<float>) + col]);
	}

}

__global__ void GetFidelityGradient(cusp::complex<float>* d_A, cusp::complex<float>* d_B, float* a, cusp::complex<float>* d_C, unsigned int lpitch, unsigned int rpitch, unsigned int imageW, unsigned int imageH)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	if (col< imageH && row < imageW)
	{
		d_A[row*lpitch / sizeof(cusp::complex<float>) + col] = (d_B[row*lpitch / sizeof(cusp::complex<float>) + col] * a[row*rpitch / sizeof(float) + col] - d_C[row*lpitch / sizeof(cusp::complex<float>) + col]) * 2.0f;
	}
}

__global__ void GetGradient(cusp::complex<float>* d_A, cusp::complex<float>* d_B, cusp::complex<float>* d_C, float wavelet_weight, float tv_weight, unsigned int imageW, unsigned int  imageH, unsigned int pitch)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	if (col< imageH && row < imageW)
	{
		d_C[row*pitch / sizeof(cusp::complex<float>) + col] += d_A[row*pitch / sizeof(cusp::complex<float>) + col] * wavelet_weight + d_B[row*pitch / sizeof(cusp::complex<float>) + col] * tv_weight;
	}
}

__global__ void complex_copy(cusp::complex<float>* d_A, const cusp::complex<float>* d_B, unsigned int width, unsigned int height)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (row < height && col < width)
	{
		d_A[row*width + col] = d_B[row*width + col];
	}

}

__global__ void depart_functor(float* d_A, float* d_B, cusp::complex<float>* d_C, unsigned int lpitch, unsigned int rpitch, unsigned int imageW, unsigned int imageH)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < imageW && row < imageH)
	{
		d_A[row*rpitch / sizeof(float) + col] = d_C[row*lpitch / sizeof(cusp::complex<float>) + col].real();
		d_B[row*rpitch / sizeof(float) + col] = d_C[row*lpitch / sizeof(cusp::complex<float>) + col].imag();
	}
}

__global__ void dot_multiply(cusp::complex<float>* d_A, cusp::complex<float>* d_B, float * d_C, cusp::complex<float>* d_D, cusp::complex<float>* d_E, unsigned int lpitch, unsigned int rpitch, unsigned int imageW, unsigned int imageH)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < imageW && row < imageH)
	{
		d_D[row*lpitch / sizeof(cusp::complex<float>) + col] = d_A[row*lpitch / sizeof(cusp::complex<float>) + col] * d_C[row*rpitch / sizeof(float) + col];
		d_E[row*lpitch / sizeof(cusp::complex<float>) + col] = d_B[row*lpitch / sizeof(cusp::complex<float>) + col] * d_C[row*rpitch / sizeof(float) + col];
	}
}

__global__ void opposite(cusp::complex<float>* d_A, cusp::complex<float>* d_B, unsigned int width, unsigned int height)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (row < height && col < width)
	{
		d_B[row*width + col] = -d_A[row*width + col];
	}
}

__global__ void GetSparseGradient(cusp::complex<float>* d_A, cusp::complex<float>* d_B, cusp::complex<float>* d_C, float epsilon, unsigned int pitch, unsigned int imageW, unsigned int imageH)
{
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < imageW && row < imageH)
	{
		d_A[row*pitch / sizeof(cusp::complex<float>) + col] /= sqrt(norm(d_A[row*pitch / sizeof(cusp::complex<float>) + col]) + epsilon);
		d_B[row*pitch / sizeof(cusp::complex<float>) + col] /= sqrt(norm(d_B[row*pitch / sizeof(cusp::complex<float>) + col]) + epsilon);
		d_C[row*pitch / sizeof(cusp::complex<float>) + col] /= sqrt(norm(d_C[row*pitch / sizeof(cusp::complex<float>) + col]) + epsilon);
	}
}

#endif //_CUDAMATRIXOP_H_
