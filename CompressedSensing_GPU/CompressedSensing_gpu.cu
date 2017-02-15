#include "stdafx.h"
#include "CompressedSensing_gpu.h"
#include <af/cuda.h>
#include <cusp/complex.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

using namespace af;

const unsigned int TILE_SIZE = 16; //32（16）
const unsigned int BLOCK_ROWS_SIZE = 16;//8（4）
const unsigned int TILE_DIM = 32; //16（32）
const unsigned int BLOCK_ROWS = 32;//4（8）
const unsigned int KERNEL_LENGTH = 4;

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

__global__ void tv_vertical(cusp::complex<float>* d_A, cusp::complex<float>* d_B, unsigned int imageW, unsigned int imageH)  //d_B为结果矩阵
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



__global__ void tv_horizontal(cusp::complex<float>* d_A, cusp::complex<float>* d_B, unsigned int imageW, unsigned int imageH)  //d_B为结果矩阵
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

__global__ void itv(cusp::complex<float>* d_A, cusp::complex<float>* d_B, cusp::complex<float>* d_C, unsigned int imageW, unsigned int imageH)   //d_C为结果矩阵
{
	__shared__ cusp::complex<float> tile1[TILE_SIZE][TILE_SIZE + 1];
	__shared__ cusp::complex<float> tile2[TILE_SIZE][TILE_SIZE + 1];
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;

	if ((y > 0 && y < imageH - 1) && (x > 0 && x < imageW - 1))
	{
		tile1[threadIdx.y][threadIdx.x] = d_A[(y - 1)*imageW + x] - d_A[y*imageW + x];
		tile2[threadIdx.y][threadIdx.x] = d_B[y*imageW + x - 1] - d_B[y*imageW + x];
	}
	if (y == 0 || x == 0)
	{
		tile1[threadIdx.y][threadIdx.x] = 0.0;
		tile2[threadIdx.y][threadIdx.x] = 0.0;
	}
	if ((y == imageH - 1) && (x == imageW - 1))
	{
		tile1[threadIdx.y][threadIdx.x] = d_A[(y - 1)*imageW + x];
		tile2[threadIdx.y][threadIdx.x] = d_B[y*imageW + x - 1];
	}

	__syncthreads();

	d_C[y*imageW + x] = tile1[threadIdx.y][threadIdx.x] + tile2[threadIdx.y][threadIdx.x];
}


CompressedSensing_gpu::CompressedSensing_gpu()
{
}
CompressedSensing_gpu::~CompressedSensing_gpu()
{
}

af::array CompressedSensing_gpu::Reconstruct(af::array & d_subsampled_data, ParameterSet & params)
{
	unsigned int height = d_subsampled_data.dims(0);
	unsigned int width = d_subsampled_data.dims(1);
	array d_kspace(height, width, c32), d_undersampled_kspace(height, width, c32), d_image(height, width, c32);
	array d_mask(height, width, f32);
	array realPart(height, width, f32);
	array imagePart(height, width, f32);
	d_undersampled_kspace(d_subsampled_data);
	d_kspace(d_subsampled_data);
	d_mask(params.mask);
	GenerateFilter(L"daub2.flt", 4);
	std::vector<float> filter = GetFilter();
	unsigned int scale = GetCoarseLevel();
	unsigned int J = QuadLength(width);
	unsigned int p = filter.size();
	std::vector<float> float_filter(filter.size());
	std::vector<float> mirror_filter(filter.size());
	auto mirror_cursor = mirror_filter.data();
	auto filter_cursor = filter.data();
	auto float_cursor = float_filter.data();
	auto end_cursor = filter_cursor + filter.size();
	for (unsigned int i = 0; i < filter.size(); ++i)
	{
		*float_cursor = static_cast<float>(*filter_cursor);
		if (i % 2 == 0)
			*mirror_cursor = *float_cursor;
		else
			*mirror_cursor = -(*float_cursor);
		++float_cursor;
		++filter_cursor;
		++mirror_cursor;
	}
	array l_Kernel(float_filter.size(), (float *)float_filter.data(), afHost);
	array h_Kernel(mirror_filter.size(), (float *)mirror_filter.data(), afHost);
	
	array d_gradient(height, width, c32), temp_image(height, width, c32), d_wavelet(height, width, c32), d_tv_vertical(height, width, c32),
		d_tv_horizontal(height, width, c32), d_new_gradient(height, width, c32), d_recon_image(height, width, c32), diff_recon_image(height, width, c32),
		d_recon_wavelet(height, width, c32), diff_recon_wavelet(height, width, c32),
		d_recon_tv_horizontal(height, width, c32), d_recon_tv_vertical(height, width, c32),
		diff_recon_tv_horizontal(height, width, c32), diff_recon_tv_vertical(height, width, c32),
		d_recon_kspace(height, width, c32),
		diff_recon_data(height, width, c32),
		diff_recon_kspace(height, width, c32),
		d_conj_module(height, width, c32),
		fft_buffer(height, width, c32),
		itv_horizontal_buffer(height, width, c32),
		itv_vertical_buffer(height, width, c32);

	array diff_recon_data_module(height, width, f32);
	array g1_norm(height, width, f32);
	array g0_norm(height, width, f32);
	array energy_buffer(height, width, f32);

	array wavelet_buffer(height, width, c32);
	array tv_buffer(height, width, c32);
	array tv_vertical_buffer(height, width, c32);
	array tv_horizontal_buffer(height, width, c32);

	VariationSet tmpSet = { d_gradient, temp_image, d_wavelet, d_tv_vertical, d_tv_horizontal, d_new_gradient, d_recon_image, diff_recon_image, d_recon_wavelet, diff_recon_wavelet,
		d_recon_tv_horizontal, d_recon_tv_vertical, diff_recon_tv_horizontal, diff_recon_tv_vertical, d_recon_kspace, diff_recon_data, diff_recon_kspace,
		d_conj_module ,diff_recon_data_module, g0_norm, g1_norm, fft_buffer, itv_horizontal_buffer, itv_vertical_buffer, wavelet_buffer,
		tv_buffer, tv_horizontal_buffer, tv_vertical_buffer, energy_buffer };

	for (unsigned int i = 0; i < 5; ++i)
	{
		Run(d_kspace, d_undersampled_kspace, d_mask, params, realPart, imagePart, scale, p, J, tmpSet, l_Kernel, h_Kernel, width, height);
	}

	return d_kspace;
}

extern "C" void CompressedSensing_gpu::Run(array& d_kspace, 
	                                 array& d_undersampled_kspace, 
	                                 array& d_mask, 
	                                 ParameterSet& params, 
	                                 af::array& realPart, 
	                                 af::array& imagPart, 
	                                 unsigned int scale, 
	                                 unsigned int p, 
	                                 unsigned int J, 
	                                 VariationSet & variationSet,
	                                 array& l_Kernel,
	                                 array& h_Kernel,
	                                 unsigned int width,
	                                 unsigned int height)
{
	_iteration_count = 0;
	float eps = (float)(2.22044604925031 * std::pow(10, -16));
	variationSet.d_image = fft2(d_kspace, height, width);
	ComputeGradient(d_kspace, params, d_mask, d_undersampled_kspace, p, 
		scale, J, variationSet, width, height, l_Kernel, h_Kernel);
	variationSet.diff_recon_data = -variationSet.d_gradient;
	float epsilon = std::pow(10, -7);

	while (true)
	{
		variationSet.d_recon_kspace = d_kspace * d_mask;
		variationSet.diff_recon_kspace = variationSet.diff_recon_data * d_mask;
		variationSet.d_recon_image = fft2(d_kspace, height, width);
		variationSet.diff_recon_image = fft2(variationSet.diff_recon_data, height, width);
		FWT2D(variationSet.d_recon_image, variationSet.d_recon_wavelet, p, scale, J, width, height, l_Kernel, h_Kernel);
		FWT2D(variationSet.diff_recon_image, variationSet.diff_recon_wavelet, p, scale, J, width, height, l_Kernel, h_Kernel);
		FTV2D(variationSet.d_recon_image, variationSet.d_tv_vertical, variationSet.d_tv_horizontal, width, height);
		FTV2D(variationSet.diff_recon_image, variationSet.diff_recon_tv_vertical, variationSet.diff_recon_tv_horizontal, width, height);

		float initial_step = 0.0f;
		float initial_energy = 0.0f;
		initial_energy = CalculateEnergy(variationSet.d_recon_kspace, variationSet.diff_recon_kspace, variationSet.d_recon_wavelet,
			variationSet.diff_recon_wavelet, variationSet.d_tv_horizontal, variationSet.d_tv_vertical, variationSet.diff_recon_tv_vertical,
			variationSet.diff_recon_tv_horizontal, d_undersampled_kspace, variationSet.energy_buffer, params, initial_step, width, height);

		float step = params.initial_line_search_step;
		float final_energy = 0.0f;
		final_energy = CalculateEnergy(variationSet.d_recon_kspace, variationSet.diff_recon_kspace, variationSet.d_recon_wavelet,
			variationSet.diff_recon_wavelet, variationSet.d_tv_horizontal, variationSet.d_tv_vertical, variationSet.diff_recon_tv_vertical,
			variationSet.diff_recon_tv_horizontal, d_undersampled_kspace, variationSet.energy_buffer, params, step, width, height);
		unsigned int line_search_times = 0;
		variationSet.d_conj_module = conjg(variationSet.d_gradient) * variationSet.diff_recon_data;

		realPart = real(variationSet.d_conj_module);
		imagPart = imag(variationSet.d_conj_module);
		float realResult = sum<float>(realPart);
		float imagResult = sum<float>(imagPart);
		float energy_variation_g0 = std::sqrt(realResult * realResult + imagResult * imagResult);

		while ((final_energy > initial_energy - params.line_search_alpha * step * energy_variation_g0) 
			&& (line_search_times < params.max_line_search_times))
		{
			line_search_times++;
			step = step * params.line_search_beta;
			final_energy = CalculateEnergy(variationSet.d_recon_kspace, variationSet.diff_recon_kspace, variationSet.d_recon_wavelet,
				variationSet.diff_recon_wavelet, variationSet.d_tv_horizontal, variationSet.d_tv_vertical, variationSet.diff_recon_tv_vertical,
				variationSet.diff_recon_tv_horizontal, d_undersampled_kspace, variationSet.energy_buffer, params, step, width, height);

		}
		if (line_search_times == params.max_line_search_times)
		{
			assert(0);
		}
		if (line_search_times > 2)
		{
			params.initial_line_search_step = params.initial_line_search_step * params.line_search_beta;
		}
		if (line_search_times < 1)
		{
			params.initial_line_search_step = params.initial_line_search_step / params.line_search_beta;
		}

		d_kspace = variationSet.diff_recon_data * step + d_kspace;
		variationSet.d_image = fft2(d_kspace, height, width);
		ComputeGradient(d_kspace, params, d_mask, d_undersampled_kspace, p, 
			scale, J, variationSet, width, height, l_Kernel, h_Kernel);
		variationSet.g0_norm = norm(variationSet.d_gradient);
		variationSet.g1_norm = norm(variationSet.d_new_gradient);
		float sum_energy_g0 = sum<float>(variationSet.g0_norm);
		float sum_energy_g1 = sum<float>(variationSet.g1_norm);
		float ellipse_factor = sum_energy_g1 / (sum_energy_g0 + eps);
		variationSet.d_gradient = variationSet.d_new_gradient;
		variationSet.diff_recon_data = variationSet.diff_recon_data * ellipse_factor - variationSet.d_new_gradient;

		++_iteration_count;
		variationSet.diff_recon_data_module = abs(variationSet.diff_recon_data);
		float differential_reconstruct_data_sum = sum<float>(variationSet.diff_recon_data_module);

		if ((_iteration_count > params.max_conjugate_gradient_iteration_times) || (differential_reconstruct_data_sum < params.gradient_tollerance))
		{
			break;
		}
	}
}


extern "C" void CompressedSensing_gpu::ComputeGradient(af::array& d_kspace, 
	                                             ParameterSet & params, 
	                                             af::array& d_mask, 
	                                             af::array& d_undersampled_kspace, 
	                                             unsigned int p, 
	                                             unsigned int scale, 
	                                             unsigned int J, 
	                                             VariationSet & variationSet,
	                                             unsigned int width,
	                                             unsigned int height,
	                                             array& l_Kernel,
	                                             array& h_Kernel)
{
	variationSet.d_gradient = (d_kspace * d_mask - d_undersampled_kspace) * 2.0f;
	float epsilon = static_cast<float>(std::pow(10, -7));
	FWT2D(variationSet.d_image, variationSet.wavelet_buffer, p, scale, J, width, height, l_Kernel, h_Kernel);
	FTV2D(variationSet.d_image, variationSet.tv_vertical_buffer, variationSet.tv_horizontal_buffer, width, height);
	if (params.tv_weight || params.wavelet_weight)
	{
		variationSet.tv_vertical_buffer /= sqrt(norm(variationSet.tv_vertical_buffer) + epsilon);
		variationSet.tv_horizontal_buffer /= sqrt(norm(variationSet.tv_horizontal_buffer) + epsilon);
		variationSet.wavelet_buffer /= sqrt(norm(variationSet.wavelet_buffer) + epsilon);
	}
	unsigned int np = static_cast<unsigned int>(std::pow(2, scale + 1));
	IFWT2D(variationSet.wavelet_buffer, variationSet.wavelet_buffer, p, scale, J, width, height, np, l_Kernel, h_Kernel);
	variationSet.wavelet_buffer = fft2(variationSet.wavelet_buffer, height, width);
	IFTV2D(variationSet.tv_vertical_buffer, variationSet.tv_horizontal_buffer, variationSet.tv_buffer, width, height);
	variationSet.tv_buffer = fft2(variationSet.tv_buffer, height, width);
	variationSet.d_gradient += variationSet.wavelet_buffer * params.wavelet_weight + variationSet.tv_buffer * params.tv_weight;
}

extern "C" float CompressedSensing_gpu::CalculateEnergy(af::array & d_recon_kspace,
	                                              af::array & diff_recon_kspace,
	                                              af::array & d_recon_wavelet,
	                                              af::array & diff_recon_wavelet,
	                                              af::array & d_tv_horizontal,
	                                              af::array & d_tv_vertical,
	                                              af::array & diff_recon_tv_vertical,
	                                              af::array & diff_recon_tv_horizontal,
	                                              af::array& d_undersampled_kspace,
	                                              af::array & energy_buffer,
	                                              ParameterSet & params,
	                                              float step,
	                                              unsigned int width,
	                                              unsigned int height)
{
	float sum_energy = 0.0f;
	array result(height, width, f32);
	result = norm(diff_recon_kspace * step + d_recon_kspace - d_undersampled_kspace);
	sum_energy += sum<float>(result);
	float epsilon = static_cast<float>(std::pow(10, -7));
	result = sqrt(norm(diff_recon_wavelet * step + d_recon_wavelet) + epsilon);
	if (params.wavelet_weight)
	{
		result = sqrt(norm(diff_recon_tv_vertical * step + d_tv_vertical) + epsilon);
		sum_energy += sum<float>(result) * params.tv_weight;
		result = sqrt(norm(diff_recon_tv_horizontal * step + d_tv_horizontal) + epsilon);
		sum_energy += sum<float>(result) * params.tv_weight;
	}

	return sum_energy;
}

extern "C" void CompressedSensing_gpu::SetFilterParams(unsigned int coarse_level)
{
	_coarse_level = coarse_level;
}

extern "C" void CompressedSensing_gpu::GenerateFilter(wchar_t * filter_type_name, unsigned int coarse_level)
{
	_filter.clear();
	assert(filter_type_name != NULL);
	SetFilterParams(coarse_level);

	std::wifstream filter_file;
	filter_file.open(filter_type_name);
	float fdata;
	if (!filter_file)
	{
		return;
	}
	if (filter_file.is_open())
	{
		while (filter_file.good() && !filter_file.eof())
		{
			filter_file >> fdata;
			_filter.push_back(fdata);
		}
	}
	filter_file.close();
}



extern "C" unsigned int CompressedSensing_gpu::QuadLength(unsigned int length)
{
	unsigned int k = 1;
	unsigned int J = 0;
	while (k < length)
	{
		k *= 2;
		J += 1;
	}
	return J;
}


extern "C" void CompressedSensing_gpu::FWT2D(array & d_raw, array& d_odata, unsigned int p, 
	unsigned int scale, unsigned int J, unsigned int width, unsigned int height, array& l_Kernel, array& h_Kernel)
{
	d_raw.eval();
	d_odata.eval();
	l_Kernel.eval();
	h_Kernel.eval();
	int af_id = af::getDevice();
	int cuda_id = afcu::getNativeId(af_id);
	cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);

	const unsigned int nc = width;
	unsigned int mem_shared;
	dim3 block;
	dim3 grid;
	block.x = nc;			                     //初始值为256
	grid.y = nc;
	unsigned int length = nc;

	dim3 dimBlock(width, 1, 1);
	dim3 dimGrid(1, height, 1);
	complex_copy << <dimGrid, dimBlock, 0, af_cuda_stream >> >((cusp::complex<float> *)d_odata.device<cfloat>(), (cusp::complex<float> *)d_raw.device<cfloat>(), width, height);

	thrust::device_vector<cusp::complex<float>>  d_tranp(nc*nc);
	for (unsigned int jscale = J - 1; jscale >= scale; --jscale)
	{
		block.x = length + p;
		block.y = 1;
		grid.x = 1;
		grid.y = length;
		mem_shared = (length + p)*sizeof(float) * 2;
		//行方向小波变换		
		DownsampleLow << <grid, block, mem_shared, af_cuda_stream>> >(p, (cusp::complex<float> *)d_odata.device<cfloat>(), raw_pointer_cast(d_tranp.data()), l_Kernel.device<float>(), length, width);
		DownsampleHigh << <grid, block, mem_shared, af_cuda_stream>> >(p, (cusp::complex<float> *)d_odata.device<cfloat>(), raw_pointer_cast(d_tranp.data()), h_Kernel.device<float>(), length, width);
		//列方向小波变换
		DownsampleLow << <grid, block, mem_shared, af_cuda_stream>> >(p, raw_pointer_cast(d_tranp.data()), (cusp::complex<float> *)d_odata.device<cfloat>(), l_Kernel.device<float>(), length, width);
		DownsampleHigh << <grid, block, mem_shared, af_cuda_stream>> >(p, raw_pointer_cast(d_tranp.data()), (cusp::complex<float> *)d_odata.device<cfloat>(), h_Kernel.device<float>(), length, width);
		length = length >> 1;
	}
	d_raw.unlock();
	d_odata.unlock();
	l_Kernel.unlock();
	h_Kernel.unlock();
}

extern "C" void CompressedSensing_gpu::IFWT2D(af::array & d_raw, array& d_odata, unsigned int p, 
	unsigned int scale, unsigned int J, unsigned int width, unsigned int height, unsigned int np, array& l_Kernel, array& h_Kernel)
{
	d_raw.eval();
	d_odata.eval();
	l_Kernel.eval();
	h_Kernel.eval();
	int af_id = af::getDevice();
	int cuda_id = afcu::getNativeId(af_id);
	cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);

	unsigned int nc = width;
	unsigned int mem_shared;
	dim3 block;
	dim3 grid;
	unsigned int length = np;
	dim3 dimBlock(width, 1, 1);
	dim3 dimGrid(1, height, 1);
	complex_copy << <dimGrid, dimBlock, 0, af_cuda_stream>> >((cusp::complex<float> *)d_odata.device<cfloat>(), (cusp::complex<float> *)d_raw.device<cfloat>(), width, height);

	thrust::device_vector<cusp::complex<float>> d_tranp(nc*nc);
	for (unsigned int jscale = scale; jscale <= J - 1; ++jscale)
	{

		mem_shared = (length + p)*sizeof(cusp::complex<float>) * 2;
		block.x = 1;
		block.y = length + p;
		grid.x = length;
		grid.y = 1;
		//上采样,行方向小波变换
		IdwtDb1D << <grid, block, mem_shared, af_cuda_stream>> >(p, (cusp::complex<float> *)d_odata.device<cfloat>(), raw_pointer_cast(d_tranp.data()), l_Kernel.device<float>(), h_Kernel.device<float>(), length, width);
		//上采样，列方向小波变换
		IdwtDb1D << <grid, block, mem_shared, af_cuda_stream>> >(p, raw_pointer_cast(d_tranp.data()), (cusp::complex<float> *)d_odata.device<cfloat>(), l_Kernel.device<float>(), h_Kernel.device<float>(), length, width);
		length = length << 1;
	}
	d_raw.unlock();
	d_odata.unlock();
	l_Kernel.unlock();
	h_Kernel.unlock();
}

extern "C" void  CompressedSensing_gpu::FTV2D(af::array & d_raw, af::array & d_vertical, af::array & d_horizontal, unsigned int width, unsigned int height)
{
	d_raw.eval();
	d_vertical.eval();
	d_horizontal.eval();
	int af_id = af::getDevice();
	int cuda_id = afcu::getNativeId(af_id);
	cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);

	dim3 dimBlock(TILE_SIZE, BLOCK_ROWS_SIZE, 1);
	dim3 dimGrid(width / TILE_SIZE, height / TILE_SIZE, 1);
	tv_vertical << <dimGrid, dimBlock, 0, af_cuda_stream>> >((cusp::complex<float> *)d_raw.device<cfloat>(), (cusp::complex<float> *)d_vertical.device<cfloat>(), width, height);
	tv_horizontal << <dimGrid, dimBlock, 0, af_cuda_stream>> >((cusp::complex<float> *)d_raw.device<cfloat>(), (cusp::complex<float> *)d_horizontal.device<cfloat>(), width, height);

	d_raw.unlock();
	d_vertical.unlock();
	d_horizontal.unlock();
}

extern "C" void CompressedSensing_gpu::IFTV2D(array& tv_vertical_buffer, array& tv_horizontal_buffer, array& tv_buffer, unsigned int width, unsigned int height)
{
	tv_vertical_buffer.eval();
	tv_horizontal_buffer.eval();
	tv_buffer.eval();
	int af_id = af::getDevice();
	int cuda_id = afcu::getNativeId(af_id);
	cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);

	dim3 dimBlock(TILE_SIZE, BLOCK_ROWS_SIZE, 1);
	dim3 dimGrid(width / TILE_SIZE, height / TILE_SIZE, 1);
	itv << <dimGrid, dimBlock, 0, af_cuda_stream>> >((cusp::complex<float> *)tv_vertical_buffer.device<cfloat>(), (cusp::complex<float> *)tv_horizontal_buffer.device<cfloat>(), (cusp::complex<float> *)tv_buffer.device<cfloat>(), width, height);

	tv_vertical_buffer.unlock();
	tv_horizontal_buffer.unlock();
	tv_buffer.unlock();
}



