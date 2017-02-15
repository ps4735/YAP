#pragma once
#include <cusp/complex.h>
#include <thrust\device_vector.h>
#include <thrust\device_ptr.h>
#include <cufft.h>
#include "ReconBase.h"

struct VariationSet
{
	thrust::device_vector<cusp::complex<float>> d_gradient;
	thrust::device_vector<cusp::complex<float>> d_image;
	thrust::device_vector<cusp::complex<float>> d_wavelet;
	thrust::device_vector<cusp::complex<float>> d_tv_vertical;
	thrust::device_vector<cusp::complex<float>> d_tv_horizontal;
	thrust::device_vector<cusp::complex<float>> d_new_gradient;
	thrust::device_vector<cusp::complex<float>> d_recon_image;
	thrust::device_vector<cusp::complex<float>> diff_recon_image;

	thrust::device_vector<cusp::complex<float>>	d_recon_wavelet;
	thrust::device_vector<cusp::complex<float>> diff_recon_wavelet;
	thrust::device_vector<cusp::complex<float>> d_recon_tv_horizontal;
	thrust::device_vector<cusp::complex<float>> d_recon_tv_vertical;
	thrust::device_vector<cusp::complex<float>>	diff_recon_tv_horizontal;
	thrust::device_vector<cusp::complex<float>> diff_recon_tv_vertical;
	thrust::device_vector<cusp::complex<float>> d_recon_kspace;
	thrust::device_vector<cusp::complex<float>> diff_recon_data;
	thrust::device_vector<cusp::complex<float>> diff_recon_kspace;
	thrust::device_vector<cusp::complex<float>> d_conj_module;

	thrust::device_vector<float> diff_recon_data_module;
	thrust::device_vector<float> g0_norm;
	thrust::device_vector<float> g1_norm;

	thrust::device_vector<cusp::complex<float>> fft_buffer;
	thrust::device_vector<cusp::complex<float>> itv_horizontal_buffer;
	thrust::device_vector<cusp::complex<float>> itv_vertical_buffer;
	thrust::device_vector<cusp::complex<float>> wavelet_buffer;
	thrust::device_vector<cusp::complex<float>> tv_buffer;
	thrust::device_vector<cusp::complex<float>> tv_horizontal_buffer;
	thrust::device_vector<cusp::complex<float>> tv_vertical_buffer;
	thrust::device_vector<float> energy_buffer;
};

class Recon_Runtime
{
public:
	Recon_Runtime();
	~Recon_Runtime();

	void Run(cusp::complex<float> *d_kspace, cusp::complex<float> *d_undersampled_kspace, float *d_mask,
		ParameterSet& params, unsigned int lpitch, unsigned int rpitch,
		bool& ft_direction, cufftHandle& plan, thrust::device_vector<float>& realPart, thrust::device_vector<float>& imagPart, unsigned int scale, unsigned int p, unsigned int J, VariationSet& variationSet);

	float CalculateEnergy(thrust::device_vector<cusp::complex<float>>& d_recon_kspace,
		thrust::device_vector<cusp::complex<float>>& diff_recon_kspace,
		thrust::device_vector<cusp::complex<float>>& d_recon_wavelet,
		thrust::device_vector<cusp::complex<float>>& diff_recon_wavelet,
		thrust::device_vector<cusp::complex<float>>& d_tv_horizontal,
		thrust::device_vector<cusp::complex<float>>& d_tv_vertical,
		thrust::device_vector<cusp::complex<float>>& diff_recon_tv_vertical,
		thrust::device_vector<cusp::complex<float>>& diff_recon_tv_horizontal,
		cusp::complex<float>* d_undersampled_kspace,
		thrust::device_vector<float> & energy_buffer,
		ParameterSet&params, float step,
		unsigned int lpitch, unsigned int rpitch);

	void ComputeGradient(cusp::complex<float> * d_kspace,
		thrust::device_vector<cusp::complex<float>>& d_gradient,
		ParameterSet& params, float * d_mask,
		cusp::complex<float> * d_undersampled_kspace,
		cufftHandle& plan,
		unsigned int lpitch, unsigned int rpitch,
		unsigned int p,
		unsigned int scale,
		unsigned int J,
		VariationSet& variationSet);
	void SetFilterParams(unsigned int filter_size);
	void GenerateFilter(wchar_t * filter_type_name, unsigned int coarse_level);
	unsigned int QuadLength(unsigned int length);
	std::vector<float> GetFilter() { return _filter; }
	unsigned int GetCoarseLevel() { return _coarse_level; }

	void GetReconData(std::vector<float> mask, const std::vector<std::complex<float>> raw_data, 
		ParameterSet& params, unsigned int width, unsigned int height, float& elapsed_time, std::complex<float> * recon_data);

protected:
	unsigned int _imageW;
	unsigned int _imageH;
	unsigned int _iteration_count;
	std::vector<float> _filter;
	unsigned int _filter_type_size;
	unsigned int _coarse_level;


};

extern "C"
{
	void Fft(cusp::complex<float>* d_idata, cusp::complex<float>* buffer, cusp::complex<float>* d_odata, bool fft_forward, cufftHandle& plan, unsigned int width, unsigned int height, unsigned int pitch);
	void FTV(cusp::complex<float>* d_A, thrust::device_vector<cusp::complex<float>>& d_B, thrust::device_vector<cusp::complex<float>>& d_C, unsigned int imageW, unsigned int imageH, unsigned int pitch);
	void IFTV(thrust::device_vector<cusp::complex<float>>& d_A, thrust::device_vector<cusp::complex<float>>& d_B, thrust::device_vector<cusp::complex<float>>& d_C, thrust::device_vector<cusp::complex<float>>& horizontal_buffer, thrust::device_vector<cusp::complex<float>>& vertical_buffer, unsigned int imageW, unsigned int imageH, unsigned int pitch);
};

