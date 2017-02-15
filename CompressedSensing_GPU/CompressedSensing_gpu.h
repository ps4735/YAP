#pragma once
#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

	struct VariationSet
	{
		af::array d_gradient;
		af::array d_image;
		af::array d_wavelet;
		af::array d_tv_vertical;
		af::array d_tv_horizontal;
		af::array d_new_gradient;
		af::array d_recon_image;
		af::array diff_recon_image;

		af::array d_recon_wavelet;
		af::array diff_recon_wavelet;
		af::array d_recon_tv_horizontal;
		af::array d_recon_tv_vertical;
		af::array diff_recon_tv_horizontal;
		af::array diff_recon_tv_vertical;
		af::array d_recon_kspace;
		af::array diff_recon_data;
		af::array diff_recon_kspace;
		af::array d_conj_module;

		af::array diff_recon_data_module;
		af::array g0_norm;
		af::array g1_norm;

		af::array fft_buffer;
	    af::array itv_horizontal_buffer;
		af::array itv_vertical_buffer;
		af::array wavelet_buffer;
		af::array tv_buffer;
		af::array tv_horizontal_buffer;
	    af::array tv_vertical_buffer;
		af::array energy_buffer;
	};

	struct ParameterSet
	{
		unsigned int max_line_search_times;
		float gradient_tollerance;
		unsigned int max_conjugate_gradient_iteration_times;
		unsigned int typenorm;
		float tv_weight;
		float wavelet_weight;
		float line_search_alpha;
		float line_search_beta;
		float initial_line_search_step;
		af::array mask;
		af::array undersampling_k_space;
	};

	
	class CompressedSensing_gpu 
	{
	public:
		CompressedSensing_gpu();
		~CompressedSensing_gpu();

		af::array Reconstruct(af::array& subsampled_data, ParameterSet& params);
		void Run(af::array& d_kspace,
			af::array& d_undersampled_kspace,
			af::array& d_mask,
			ParameterSet& params,
			af::array& realPart,
			af::array& imagPart,
			unsigned int scale,
			unsigned int p,
			unsigned int J,
			VariationSet& variationSet,
			af::array& l_Kernel,
			af::array& h_Kernel,
			unsigned int width,
			unsigned int height);

		float CalculateEnergy(af::array& d_recon_kspace,
			                  af::array& diff_recon_kspace,
			                  af::array& d_recon_wavelet,
			                  af::array& diff_recon_wavelet,
			                  af::array& d_tv_horizontal,
			                  af::array& d_tv_vertical,
			                  af::array& diff_recon_tv_vertical,
			                  af::array& diff_recon_tv_horizontal,
			                  af::array& d_undersampled_kspace,
			                  af::array& energy_buffer,
			                  ParameterSet&params, 
			                  float step,
			                  unsigned int width,
			                  unsigned int height);


		void ComputeGradient(af::array& d_kspace,
			ParameterSet& params,
			af::array& d_mask,
			af::array& d_undersampled_kspace,
			unsigned int p,
			unsigned int scale,
			unsigned int J,
			VariationSet& variationSet,
			unsigned int width,
			unsigned int height,
			af::array& l_Kernel,
			af::array& h_Kernel);

		void SetFilterParams(unsigned int filter_size);
		void GenerateFilter(wchar_t * filter_type_name, unsigned int coarse_level);
		unsigned int QuadLength(unsigned int length);
		std::vector<float> GetFilter() { return _filter; }
		unsigned int GetCoarseLevel() { return _coarse_level; }
		void FWT2D(af::array& d_raw, af::array& d_odata, unsigned int p, unsigned int scale, unsigned int J, unsigned int width, unsigned int height, af::array& l_Kernel, af::array& h_Kernel);
		void IFWT2D(af::array& d_raw, af::array& d_odata, unsigned int p, unsigned int scale, unsigned int J, unsigned int width, unsigned int height, unsigned int np, af::array& l_Kernel, af::array& h_Kernel);
		void FTV2D(af::array & d_raw, af::array & d_vertical, af::array & d_horizontal, unsigned int width, unsigned int height);
		void IFTV2D(af::array& tv_vertical_buffer, af::array& tv_horizontal_buffer, af::array& tv_buffer, unsigned int width, unsigned int height);
	private:
		std::vector<float> _filter;
		unsigned int _filter_type_size;
		unsigned int _coarse_level;
		unsigned int _iteration_count;

	};
