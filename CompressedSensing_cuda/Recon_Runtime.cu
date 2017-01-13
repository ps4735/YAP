#include "stdafx.h"
#include "Recon_Runtime.h"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "cudaMatrixOp.cuh"
#include <thrust\device_ptr.h>
#include <thrust\transform.h>
#include <thrust\functional.h>
#include <thrust\reduce.h>

using namespace  cusp;
using namespace  thrust; 

Recon_Runtime::Recon_Runtime()
{
}


Recon_Runtime::~Recon_Runtime()
{
}

//仿函数定义
struct saxpy_functor
{
	const float a;
	saxpy_functor(float _a) : a(_a) {};
	__host__ __device__
		complex<float> operator() (const complex<float> & x, const complex<float> & y) const {
		return (x * a + y);
	}
};

struct saxmy_functor
{
	const float a;
	saxmy_functor(float _a) : a(_a) {};
	__host__ __device__
		complex<float> operator() (const complex<float> & x, const complex<float> & y) const {
		return x * a - y;
	}
};

struct sparse_energy
{
	const float a;
	const float b;
	sparse_energy(float _a, float _b) : a(_a), b(_b) {};
	__host__ __device__
		float operator() (const complex<float> & x, const complex<float> & y) const {
		return sqrt(norm(x*a + y) + b);
	}
};

struct sparse_gradient
{
	const float a;
	sparse_gradient(float _a) : a(_a) {};
	__host__ __device__
		complex<float> operator() (const complex<float> & x) const {
		return x / sqrt(norm(x) + a);
	}
};

struct module_functor
{
	__host__ __device__
		float operator() (const complex<float> & x) const {
		return abs(x);
	}
};

//复数矩阵的共轭矩阵乘以复数矩阵
struct  conj_multiply
{
	__host__ __device__
		complex<float> operator() (const complex<float> & x, const complex<float> & y) const {
		return conj(x) * y;
	}
};

//Set kernel
extern "C" void setConvolutionLowKernel(float *filter)
{
	cudaMemcpyToSymbol(l_Kernel, filter, sizeof(float) * KERNEL_LENGTH);
}

extern "C" void setConvolutionHighKernel(float *filter)
{
	cudaMemcpyToSymbol(h_Kernel, filter, sizeof(float) * KERNEL_LENGTH);
}

extern "C" void Recon_Runtime::SetFilterParams(unsigned int coarse_level)
{
	_coarse_level = coarse_level;
}

extern "C" void Recon_Runtime::GenerateFilter(wchar_t * filter_type_name, unsigned int coarse_level)
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

extern "C" unsigned int Recon_Runtime::QuadLength(unsigned int length)
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



extern "C" void Fwt2D(complex<float>* d_raw, complex<float>* d_odata, unsigned int pitch, unsigned int p, unsigned int scale, unsigned int J, unsigned int imageW, unsigned int imageH)
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
	complex_copy << <dimGrid, dimBlock >> >(d_odata, d_raw, imageW, imageH);

	device_vector<complex<float>>  d_tranp(nc*nc);
	for (unsigned int jscale = J - 1; jscale >= scale; --jscale)
	{
		block.x = length + p;
		block.y = 1;
		grid.x = 1;
		grid.y = length;
		mem_shared = (length + p)*sizeof(float) * 2;
		//行方向小波变换		
		DownsampleLow << <grid, block, mem_shared >> >(p, d_odata, raw_pointer_cast(d_tranp.data()), length, pitch);
		DownsampleHigh << <grid, block, mem_shared >> >(p, d_odata, raw_pointer_cast(d_tranp.data()), length, pitch);
		//列方向小波变换
		DownsampleLow << <grid, block, mem_shared >> >(p, raw_pointer_cast(d_tranp.data()), d_odata, length, pitch);
		DownsampleHigh << <grid, block, mem_shared >> >(p, raw_pointer_cast(d_tranp.data()), d_odata, length, pitch);
		length = length >> 1;
	}
}

extern "C" void IFwt2D(complex<float>* d_raw, complex<float>* d_odata, unsigned int pitch, unsigned int p, unsigned int scale, unsigned int J, unsigned int imageW, unsigned int imageH, unsigned int np)
{
	unsigned int nc = imageW;
	unsigned int mem_shared;
	//copy kernel 配置
	dim3 block;
	dim3 grid;
	unsigned int length = np;

	dim3 dimBlock(imageW, 1, 1);
	dim3 dimGrid(1, imageH, 1);
	complex_copy << <dimGrid, dimBlock >> >(d_odata, d_raw, imageW, imageH);

	device_vector<complex<float>> d_tranp(nc*nc);
	for (unsigned int jscale = scale; jscale <= J - 1; ++jscale)
	{

		mem_shared = (length + p)*sizeof(complex<float>) * 2;
		block.x = 1;
		block.y = length + p;
		grid.x = length;
		grid.y = 1;
		//上采样,行方向小波变换
		IdwtDb1D << <grid, block, mem_shared >> >(p, d_odata, raw_pointer_cast(d_tranp.data()), length, pitch);
		//上采样，列方向小波变换
		IdwtDb1D << <grid, block, mem_shared >> >(p, raw_pointer_cast(d_tranp.data()), d_odata, length, pitch);
		length = length << 1;
	}
}

extern "C" void Recon_Runtime::ComputeGradient(complex<float> * d_kspace,
	device_vector<complex<float>>& d_gradient,
	ParameterSet& params, float * d_mask,
	complex<float> * d_undersampled_kspace,
	cufftHandle& plan,
	unsigned int lpitch, unsigned int rpitch,
	unsigned int p,
	unsigned int scale,
	unsigned int J,
	VariationSet& variationSet)
{
	dim3 dimBlock(_imageW, 1, 1);
	dim3 dimGrid(1, _imageH, 1);

	GetFidelityGradient << <dimGrid, dimBlock >> >(raw_pointer_cast(d_gradient.data()), d_kspace, d_mask, d_undersampled_kspace, lpitch, rpitch, _imageW, _imageH);		//原位运算，初始值为d_kspace


	float epsilon = static_cast<float>(std::pow(10, -7));

	cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	Fwt2D(raw_pointer_cast(variationSet.d_image.data()), raw_pointer_cast(variationSet.wavelet_buffer.data()), lpitch, p, scale, J, _imageW, _imageH);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float   elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	FTV(raw_pointer_cast(variationSet.d_image.data()), variationSet.tv_vertical_buffer, variationSet.tv_horizontal_buffer, _imageW, _imageH, lpitch);

	if (params.tv_weight || params.wavelet_weight)
	{
		GetSparseGradient << <dimGrid, dimBlock >> >(raw_pointer_cast(variationSet.tv_vertical_buffer.data()), raw_pointer_cast(variationSet.tv_horizontal_buffer.data()), raw_pointer_cast(variationSet.wavelet_buffer.data()), epsilon, lpitch, _imageW, _imageH);//原位运算
	}


	unsigned int np = static_cast<unsigned int>(std::pow(2, scale + 1));

	IFwt2D(raw_pointer_cast(variationSet.wavelet_buffer.data()), raw_pointer_cast(variationSet.wavelet_buffer.data()), lpitch, p, scale, J, _imageW, _imageH, np);							//原位运算

	bool ft_direction = true;
	Fft(raw_pointer_cast(variationSet.wavelet_buffer.data()), raw_pointer_cast(variationSet.fft_buffer.data()), raw_pointer_cast(variationSet.wavelet_buffer.data()), ft_direction, plan, _imageW, _imageH, lpitch);				//原位运算
	IFTV(variationSet.tv_vertical_buffer, variationSet.tv_horizontal_buffer, variationSet.tv_buffer, variationSet.itv_horizontal_buffer, variationSet.itv_vertical_buffer, _imageW, _imageH, lpitch);

	Fft(raw_pointer_cast(variationSet.tv_buffer.data()), raw_pointer_cast(variationSet.fft_buffer.data()), raw_pointer_cast(variationSet.tv_buffer.data()), ft_direction, plan, _imageW, _imageH, lpitch);							//原位运算

	GetGradient << <dimGrid, dimBlock >> >(raw_pointer_cast(variationSet.wavelet_buffer.data()), raw_pointer_cast(variationSet.tv_buffer.data()), raw_pointer_cast(d_gradient.data()), params.wavelet_weight, params.tv_weight, _imageW, _imageH, lpitch);

}

extern "C" float Recon_Runtime::CalculateEnergy(device_vector<complex<float>>& d_recon_kspace,
	device_vector<complex<float>>& diff_recon_kspace,
	device_vector<complex<float>>& d_recon_wavelet,
	device_vector<complex<float>>& diff_recon_wavelet,
	device_vector<complex<float>>& d_tv_horizontal,
	device_vector<complex<float>>& d_tv_vertical,
	device_vector<complex<float>>& diff_recon_tv_vertical,
	device_vector<complex<float>>& diff_recon_tv_horizontal,
	complex<float>* d_undersampled_kspace,
	device_vector<float> & energy_buffer,
	ParameterSet&params, float step,
	unsigned int lpitch, unsigned int rpitch)
{
	float sum_energy = 0.0f;

	dim3 dimGrid(1, _imageH, 1);
	dim3 dimBlock(_imageW, 1, 1);

	//分配空间	
	device_vector<float> result(_imageH*_imageW);

	CalculateFidelity << <dimGrid, dimBlock >> >(raw_pointer_cast(diff_recon_kspace.data()), raw_pointer_cast(d_recon_kspace.data()), step, d_undersampled_kspace, raw_pointer_cast(result.data()), lpitch, rpitch, _imageW, _imageH);  //三元运算
	sum_energy += thrust::reduce(result.begin(), result.end(), 0.0f, thrust::plus<float>());

	float epsilon = static_cast<float>(std::pow(10, -7));
	//计算小波项
	transform(diff_recon_wavelet.begin(), diff_recon_wavelet.end(), d_recon_wavelet.begin(), result.begin(), sparse_energy(step, epsilon));   //二元运算
	if (params.wavelet_weight)
	{
		sum_energy += thrust::reduce(result.begin(), result.end(), 0.0f, thrust::plus<float>()) * params.wavelet_weight;
	}
	// 计算TV项能量值
	if (params.tv_weight)
	{
		transform(diff_recon_tv_vertical.begin(), diff_recon_tv_vertical.end(), d_tv_vertical.begin(), result.begin(), sparse_energy(step, epsilon)); //二元运算
		sum_energy += thrust::reduce(result.begin(), result.end(), 0.0f, thrust::plus<float>()) * params.tv_weight;
		transform(diff_recon_tv_horizontal.begin(), diff_recon_tv_horizontal.end(), d_tv_horizontal.begin(), result.begin(), sparse_energy(step, epsilon));  //二元运算
		sum_energy += thrust::reduce(result.begin(), result.end(), 0.0f, thrust::plus<float>()) * params.tv_weight;
	}
	return sum_energy;
}

extern "C" void Recon_Runtime::GetReconData(std::vector<float> mask, std::vector<std::complex<float>> raw_data, 
	ParameterSet& params, unsigned int width, unsigned int height, float& elapsed_time, std::complex<float> * recon_data)
{
	_imageW = width;
	_imageH = height;

	complex<float> *d_kspace, *d_undersampled_kspace, *d_image;
	float *d_mask;

	size_t lpitch, rpitch;						//lpitch代表元素为复数类的对齐参数， rpitch代表元素为实数类的对齐参数

	dim3 dimBlock(_imageW, 1, 1);
	dim3 dimGrid(1, _imageH, 1);

	device_vector<float> realPart(_imageH*_imageW);
	device_vector<float> imagPart(_imageH*_imageW);


	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	cudaEventRecord(start, 0);

	//制定fft plan和方向
	bool ft_direction = false;							//false为傅立叶逆变换， 从k空间到图像域; true为傅立叶正变换，从图像域到k空间
	cufftHandle plan;
	checkCudaErrors(cufftPlan2d(&plan, _imageH, _imageW, CUFFT_C2C));

	checkCudaErrors(cudaMallocPitch((void **)&d_kspace, &lpitch, _imageW*sizeof(complex<float>), _imageH));
	checkCudaErrors(cudaMallocPitch((void **)&d_undersampled_kspace, &lpitch, _imageW*sizeof(complex<float>), _imageH));
	checkCudaErrors(cudaMallocPitch((void **)&d_mask, &rpitch, _imageW*sizeof(float), _imageH));
	checkCudaErrors(cudaMallocPitch((void **)&d_image, &lpitch, _imageW*sizeof(complex<float>), _imageH));


	checkCudaErrors(cudaMemcpy2D(d_kspace, lpitch, raw_data.data(), sizeof(float) * 2 * _imageW, sizeof(float) * 2 * _imageW, _imageH, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy2D(d_mask, rpitch, mask.data(), sizeof(float)*_imageW, sizeof(float)*_imageW, _imageH, cudaMemcpyHostToDevice));
	complex_copy << <dimGrid, dimBlock >> >(d_undersampled_kspace, d_kspace, _imageW, _imageH);


	//小波变换的掩模
//	std::shared_ptr<CWavelet> wavelet = std::shared_ptr<CWavelet>(new CWavelet());
	GenerateFilter(L"daub2.flt", 4);
	std::vector<float> filter = GetFilter();
	unsigned int scale = GetCoarseLevel();
	unsigned int J = QuadLength(_imageW);
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

	//将小波变换基拷贝到设备端
	setConvolutionLowKernel(float_filter.data());
	setConvolutionHighKernel(mirror_filter.data());
	//定义临时变量
	device_vector<complex<float>> d_gradient(_imageH*_imageW), temp_image(_imageH*_imageW), d_wavelet(_imageH*_imageW), d_tv_vertical(_imageH*_imageW),
		d_tv_horizontal(_imageH*_imageW), d_new_gradient(_imageH*_imageW), d_recon_image(_imageH*_imageW), diff_recon_image(_imageH*_imageW),
		d_recon_wavelet(_imageH*_imageW), diff_recon_wavelet(_imageH*_imageW),
		d_recon_tv_horizontal(_imageH*_imageW), d_recon_tv_vertical(_imageH*_imageW),
		diff_recon_tv_horizontal(_imageH*_imageW), diff_recon_tv_vertical(_imageH*_imageW),
		d_recon_kspace(_imageH*_imageW),
		diff_recon_data(_imageH*_imageW),
		diff_recon_kspace(_imageW*_imageH),
		d_conj_module(_imageW*_imageH),
		fft_buffer(_imageW*_imageH),
		itv_horizontal_buffer(_imageW*_imageH),
		itv_vertical_buffer(_imageW*_imageH);

	device_vector<float> diff_recon_data_module(_imageW*_imageH);
	device_vector<float> g1_norm(_imageH*_imageW);
	device_vector<float> g0_norm(_imageW*_imageH);
	device_vector<float> energy_buffer(_imageW*_imageH);

	device_vector<complex<float>> wavelet_buffer(_imageW*_imageH);
	device_vector<complex<float>> tv_buffer(_imageW*_imageH);
	device_vector<complex<float>> tv_vertical_buffer(_imageW*_imageH);
	device_vector<complex<float>> tv_horizontal_buffer(_imageW*_imageH);


	VariationSet tmpSet = { d_gradient, temp_image, d_wavelet, d_tv_vertical, d_tv_horizontal, d_new_gradient, d_recon_image, diff_recon_image, d_recon_wavelet, diff_recon_wavelet,
		d_recon_tv_horizontal, d_recon_tv_vertical, diff_recon_tv_horizontal, diff_recon_tv_vertical, d_recon_kspace, diff_recon_data, diff_recon_kspace,
		d_conj_module ,diff_recon_data_module, g0_norm, g1_norm, fft_buffer, itv_horizontal_buffer, itv_vertical_buffer, wavelet_buffer,
		tv_buffer, tv_horizontal_buffer, tv_vertical_buffer, energy_buffer };
	//重建迭代
//	for (int i = 0; i < 5; ++i)
//	{
		Run(d_kspace, d_undersampled_kspace, d_mask,
			params, lpitch, rpitch,
			ft_direction, plan, realPart, imagPart, scale, p, J, tmpSet);
//	}


//	Fft(d_kspace, raw_pointer_cast(tmpSet.fft_buffer.data()), d_image, ft_direction, plan, _imageW, _imageH, lpitch);

	checkCudaErrors(cudaMemcpy2D(recon_data, sizeof(float)*_imageW * 2, d_kspace, lpitch, sizeof(float)*_imageW * 2, _imageH, cudaMemcpyDeviceToHost));	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
// 	auto cursor = recon_data;
// 	for (auto iter = raw_data.begin(); iter != raw_data.end(); ++iter)
// 	{
// 		float real = static_cast<float>((*iter).real());
// 		float imag = static_cast<float>((*iter).imag());
// 		*cursor = std::complex<float>(real, imag);
// 		++cursor;
// 	}

	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cudaFree(d_kspace));
	checkCudaErrors(cudaFree(d_mask));
	checkCudaErrors(cudaFree(d_undersampled_kspace));
	checkCudaErrors(cudaFree(d_image));


}

extern "C" void Recon_Runtime::Run(complex<float> *d_kspace, complex<float> *d_undersampled_kspace, float *d_mask,
	ParameterSet& params, unsigned int lpitch, unsigned int rpitch,
	bool& ft_direction, cufftHandle& plan, device_vector<float>& realPart, device_vector<float>& imagPart, unsigned int p, unsigned int scale, unsigned int J, VariationSet& variationSet)
{
	_iteration_count = 0;
	float eps = (float)(2.22044604925031 * std::pow(10, -16));
	//kernel配置
	dim3 dimBlock(_imageW, 1, 1);
	dim3 dimGrid(1, _imageH, 1);

	device_ptr<complex<float>> kspace_ptr(d_kspace);
	Fft(d_kspace, raw_pointer_cast(variationSet.fft_buffer.data()), raw_pointer_cast(variationSet.d_image.data()), ft_direction, plan, _imageW, _imageH, lpitch);				//由k空间数据得到初始图像 [非原位运算]

	ComputeGradient(d_kspace, variationSet.d_gradient, params, d_mask, d_undersampled_kspace, plan, lpitch, rpitch, p, scale, J, variationSet);			      //求得g0
																																						  //transform(variationSet.d_gradient.begin(), variationSet.d_gradient.end(), variationSet.diff_recon_data.begin(), opposite_functor());                    //diff_recon_data = -g0，得到dx的初始值
	opposite << <dimGrid, dimBlock >> >(raw_pointer_cast(variationSet.d_gradient.data()), raw_pointer_cast(variationSet.diff_recon_data.data()), _imageW, _imageH);

	float epsilon = std::pow(10, -7);
	//重建过程
	while (true)
	{
		// diff_recon_kspace = -g0 * mask; 
		dot_multiply << <dimGrid, dimBlock >> >(d_kspace, raw_pointer_cast(variationSet.diff_recon_data.data()), d_mask, raw_pointer_cast(variationSet.d_recon_kspace.data()), raw_pointer_cast(variationSet.diff_recon_kspace.data()), lpitch, rpitch, _imageW, _imageH);

		Fft(d_kspace, raw_pointer_cast(variationSet.fft_buffer.data()), raw_pointer_cast(variationSet.d_recon_image.data()), ft_direction, plan, _imageW, _imageH, lpitch);													 //临时变量
		Fft(raw_pointer_cast(variationSet.diff_recon_data.data()), raw_pointer_cast(variationSet.fft_buffer.data()), raw_pointer_cast(variationSet.diff_recon_image.data()), ft_direction, plan, _imageW, _imageH, lpitch);    //临时变量

		Fwt2D(raw_pointer_cast(variationSet.d_recon_image.data()), raw_pointer_cast(variationSet.d_recon_wavelet.data()), lpitch, p, scale, J, _imageW, _imageH);			   //临时变量
		Fwt2D(raw_pointer_cast(variationSet.diff_recon_image.data()), raw_pointer_cast(variationSet.diff_recon_wavelet.data()), lpitch, p, scale, J, _imageW, _imageH);	   //临时变量

		FTV(raw_pointer_cast(variationSet.d_recon_image.data()), variationSet.d_tv_vertical, variationSet.d_tv_horizontal, _imageW, _imageH, lpitch);                                      //无临时变量
		FTV(raw_pointer_cast(variationSet.diff_recon_image.data()), variationSet.diff_recon_tv_vertical, variationSet.diff_recon_tv_horizontal, _imageW, _imageH, lpitch);                 //无临时变量


		float initial_step = 0.0f;
		float initial_energy = 0.0f;

		initial_energy = CalculateEnergy(variationSet.d_recon_kspace,
			variationSet.diff_recon_kspace,
			variationSet.d_recon_wavelet, variationSet.diff_recon_wavelet,
			variationSet.d_tv_horizontal, variationSet.d_tv_vertical,
			variationSet.diff_recon_tv_vertical, variationSet.diff_recon_tv_horizontal,
			d_undersampled_kspace,
			variationSet.energy_buffer,
			params, initial_step, lpitch, rpitch);

		float step = params.initial_line_search_step;


		float final_energy = 0.0f;
		final_energy = CalculateEnergy(variationSet.d_recon_kspace,
			variationSet.diff_recon_kspace,
			variationSet.d_recon_wavelet, variationSet.diff_recon_wavelet,
			variationSet.d_tv_horizontal, variationSet.d_tv_vertical,
			variationSet.diff_recon_tv_vertical, variationSet.diff_recon_tv_horizontal,
			d_undersampled_kspace,
			variationSet.energy_buffer,
			params, step, lpitch, rpitch);

		unsigned int line_search_times = 0;

		transform(variationSet.d_gradient.begin(), variationSet.d_gradient.end(), variationSet.diff_recon_data.begin(), variationSet.d_conj_module.begin(), conj_multiply());

		//求搜索最优的梯度方向
		depart_functor << <dimGrid, dimBlock >> >(raw_pointer_cast(realPart.data()), raw_pointer_cast(imagPart.data()), raw_pointer_cast(variationSet.d_conj_module.data()), lpitch, rpitch, _imageW, _imageH);
		float realResult = thrust::reduce(realPart.begin(), realPart.end(), 0.0f, thrust::plus<float>());
		float imagResult = thrust::reduce(imagPart.begin(), imagPart.end(), 0.0f, thrust::plus<float>());
		float energy_variation_g0 = std::sqrt(realResult*realResult + imagResult*imagResult);

		//线搜索
		while ((final_energy > initial_energy - params.line_search_alpha * step * energy_variation_g0) &&
			(line_search_times < params.max_line_search_times))
		{
			line_search_times++;
			step = step * params.line_search_beta;
			final_energy = CalculateEnergy(variationSet.d_recon_kspace,
				variationSet.diff_recon_kspace,
				variationSet.d_recon_wavelet, variationSet.diff_recon_wavelet,
				variationSet.d_tv_horizontal, variationSet.d_tv_vertical,
				variationSet.diff_recon_tv_vertical, variationSet.diff_recon_tv_horizontal,
				d_undersampled_kspace,
				variationSet.energy_buffer,
				params, step, lpitch, rpitch);
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

		transform(variationSet.diff_recon_data.begin(), variationSet.diff_recon_data.end(), kspace_ptr, kspace_ptr, saxpy_functor(step));      //更新d_kspace 

		Fft(d_kspace, raw_pointer_cast(variationSet.fft_buffer.data()), raw_pointer_cast(variationSet.d_image.data()), ft_direction, plan, _imageW, _imageH, lpitch);     //d_kspace更新，d_image更新

		ComputeGradient(d_kspace, variationSet.d_new_gradient, params, d_mask, d_undersampled_kspace, plan, lpitch, rpitch, p, scale, J, variationSet);				////更新d_new_gradient, g1

		norm_kernel << <dimGrid, dimBlock >> >(raw_pointer_cast(variationSet.d_gradient.data()), raw_pointer_cast(variationSet.d_new_gradient.data()), raw_pointer_cast(variationSet.g0_norm.data()), raw_pointer_cast(variationSet.g1_norm.data()), lpitch, rpitch, _imageW, _imageH);

		float sum_energy_g0 = thrust::reduce(variationSet.g0_norm.begin(), variationSet.g0_norm.end(), 0.0f, thrust::plus<float>());
		float sum_energy_g1 = thrust::reduce(variationSet.g1_norm.begin(), variationSet.g1_norm.end(), 0.0f, thrust::plus<float>());
		float ellipse_factor = sum_energy_g1 / (sum_energy_g0 + eps);

		complex_copy << <dimGrid, dimBlock >> >(raw_pointer_cast(variationSet.d_gradient.data()), raw_pointer_cast(variationSet.d_new_gradient.data()), _imageW, _imageH);    //g0 = g1 更新g0

		transform(variationSet.diff_recon_data.begin(), variationSet.diff_recon_data.end(), variationSet.d_new_gradient.begin(), variationSet.diff_recon_data.begin(), saxmy_functor(ellipse_factor));             //更新diff_recon_data;

		++_iteration_count;

		transform(variationSet.diff_recon_data.begin(), variationSet.diff_recon_data.end(), variationSet.diff_recon_data_module.begin(), module_functor());
		float differential_reconstruct_data_sum = thrust::reduce(variationSet.diff_recon_data_module.begin(), variationSet.diff_recon_data_module.end(), 0.0f, thrust::plus<float>());

		if ((_iteration_count > params.max_conjugate_gradient_iteration_times) ||
			(differential_reconstruct_data_sum < params.gradient_tollerance))
		{
			break;
		}
	}

}