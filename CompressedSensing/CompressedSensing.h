#pragma once
#include "Interface/Implement/ProcessorImpl.h"
#include "fftw3.h"
#include <complex>
#include <armadillo>


namespace Yap
{
	struct ParametterSet
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
		arma::fmat mask;
		arma::cx_fmat undersampling_k_space;
	};

	struct  ILongTaskListener
	{
		virtual void OnBegin() = 0;
		virtual bool OnProgress(unsigned int percent) = 0;
		virtual void OnEnd() = 0;
	};

	class CompressedSensing :
		public ProcessorImpl
	{
	public:
		CompressedSensing();
		virtual ~CompressedSensing();

		virtual IProcessor * Clone() override;
		virtual bool Input(const wchar_t * name, IData * data) override;

	protected:
		SmartPtr<IData> _mask;

		arma::cx_fmat Reconstruct(arma::cx_fmat& subsampled_data, ParametterSet& myparameter);

		arma::cx_fmat ComputeGradient(arma::cx_fmat &in_data, float wavelet_weight, float tv_weight,
			float p_norm, arma::fmat mask, arma::cx_fmat subsampled_kdata);

		arma::cx_fmat GetFidelityTerm(arma::cx_fmat& in_data, arma::fmat& mask, arma::cx_fmat subsampled_kdata);
		arma::cx_fmat GetWaveletTerm(arma::cx_fmat& in_data, float p_norm);
		arma::cx_fmat GetTVTerm(arma::cx_fmat& in_data, float p_norm);
		float CalculateEnergy(arma::cx_fmat& recon_k_data, arma::cx_fmat& differential_recon_kdata,
			arma::cx_fmat& recon_wavelet_data, arma::cx_fmat& differential_recon_wavelet_data, 
			arma::cx_fcube& recon_tv_data, arma::cx_fcube& differential_recon_tv_data, 
			ParametterSet& myparameter, float step_length);


		//Fwt
		unsigned int QuadLength(unsigned int length);
		std::vector<float> Iconv(std::vector<float>& filter, std::vector<float>& row);
		arma::cx_fmat Fw2DTransform(arma::cx_fmat& input, ILongTaskListener * listener = nullptr);
		arma::fmat FWT2D(arma::fmat& input, std::vector<float>& filter, unsigned int scale, ILongTaskListener * listener = nullptr);
		std::vector<float> Filter(std::vector<float>& filter, unsigned int a, std::vector<float>& in_put);
		std::vector<float> MirrorFilt(std::vector<float>& filter);
		std::vector<float> Aconv(std::vector<float>& filter, std::vector<float>& row);
		void SetFilterParams(unsigned int filter_size);
		void GenerateFilter(wchar_t * filter_type_name, unsigned int coarse_level);
		void SetFilter(float * filter, unsigned int size);
		std::vector<float> DownSamplingLowPass(std::vector<float>& row, std::vector<float>& filter);
		std::vector<float> DownSamplingHighPass(std::vector<float>& row, std::vector<float>& filter);
		std::vector<float> LeftShift(std::vector<float>& row);
		arma::fmat DownSampling(arma::fmat& output, std::vector<float>& filter, unsigned int nc);

		//Ifwt
		arma::cx_fmat IFw2DTransform(arma::cx_fmat& input, ILongTaskListener * listener = nullptr);
		arma::fmat IFWT2D(arma::fmat& input, std::vector<float>& filter, unsigned int scale, ILongTaskListener * listener = nullptr);
		arma::fmat UpSampling(arma::fmat& input, std::vector<float>& filter, unsigned int nc);
		std::vector<float> UpSamplingLowPass(std::vector<float>& row, std::vector<float>& filter);
		std::vector<float> UpSampleInterpolateZero(std::vector<float>& row);
		std::vector<float> UpSamplingHighPass(std::vector<float>& row, std::vector<float>& filter);
		std::vector<float> RightShift(std::vector<float>& row);

		//TV
		arma::cx_fcube TV2DTransform(arma::cx_fmat& in_data);
		//ITV2D
		arma::cx_fmat ITV2DTransform(arma::cx_fcube& input);

		//Fft
// 		void Plan(size_t width, size_t height, bool inverse, bool inplace);
// 		void FftShift(boost::numeric::ublas::matrix<std::complex<float>>& data);
// 		void SwapBlock(std::complex<float> * block1, std::complex<float> * block2,
// 			size_t width, size_t height, size_t line_stride);

		arma::fmat Transpose(arma::fmat& in_data);
		arma::cx_fmat Transpose(arma::cx_fmat& in_data);
		arma::fmat square_module(arma::cx_fmat& input);
//		boost::numeric::ublas::matrix<float> fill(float value, boost::numeric::ublas::matrix<float>& input);
		arma::fmat sqrt_root(arma::fmat &input);
		arma::cx_fmat conj_multiply(arma::cx_fmat& input_1, arma::cx_fmat& input_2);
		arma::fmat module(arma::cx_fmat& input);
		arma::cx_fmat dot_multiply(arma::cx_fmat in_data, arma::fmat mask);


	private:
		std::vector<float> _filter;
		unsigned int _filter_type_size;
		unsigned int _coarse_level;
		unsigned int _iteration_count;
// 		unsigned int _plan_data_width;
// 		unsigned int _plan_data_height;
// 		bool _plan_inverse;
// 		bool _plan_in_place;

//		fftwf_plan _fft_plan;
	};
}

