#include "stdafx.h"
#include "CompressedSensing.h"
#include "Interface/Client/DataHelper.h"
#include "Interface/Implement/DataObject.h"
#include "vector"
#include <algorithm>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Yap;
using namespace arma;


CompressedSensing::CompressedSensing():
	ProcessorImpl(L"CompressedSensing"),
	_iteration_count(0),
	_coarse_level(0),
	_filter_type_size(0)
{
	AddInput(L"Input", YAP_ANY_DIMENSION, DataTypeComplexFloat);
	AddInput(L"Mask", 2, DataTypeFloat);
	AddOutput(L"Output", 2, DataTypeComplexFloat);
}


CompressedSensing::~CompressedSensing()
{
}


IProcessor * Yap::CompressedSensing::Clone()
{
	return new (nothrow) CompressedSensing(*this);
}

bool Yap::CompressedSensing::Input(const wchar_t * port, IData * data)
{	
	if (wstring(port) == L"Mask")
	{
		_mask = YapShared(data);
	}
	else if (wstring(port) == L"Input")
	{
		if (!_mask)
			return false;
		DataHelper input_data(data);
		DataHelper input_mask(_mask.get());
		fmat mask(input_mask.GetHeight(), input_mask.GetWidth());
		cx_fmat undersampled_data(input_data.GetHeight(), input_data.GetWidth());

 		memcpy(mask.memptr(), GetDataArray<float>(_mask.get()), input_data.GetWidth() * input_data.GetHeight() * sizeof(float));
 		memcpy(undersampled_data.memptr(), GetDataArray<complex<float>>(data), input_data.GetWidth() * input_data.GetHeight() * sizeof(complex<float>));
		ParametterSet myparameter = { 200, float(pow(10,-30)), 8, 1, 0.002f, 0.005f, 0.01f, 0.8f, 2, mask, undersampled_data};

		auto recon_data = Reconstruct(undersampled_data, myparameter);
		std::complex<float> * recon = nullptr;
		try
		{
			recon = new complex<float>[input_data.GetWidth() * input_data.GetHeight()];
		}
		catch (bad_alloc&)
		{
			return false;
		}
		memcpy(recon, recon_data.memptr(), input_data.GetWidth() * input_data.GetHeight() * sizeof(complex<float>));
		Dimensions dimensions;
		dimensions(DimensionReadout, 0U, input_data.GetWidth())
			(DimensionPhaseEncoding, 0U, input_data.GetHeight());
		auto output = YapShared(new ComplexFloatData(recon, dimensions, nullptr, true));

		Feed(L"Output", output.get());
	}
	else
	{
		return false;
	}
  return true;
}

cx_fmat Yap::CompressedSensing::Reconstruct(cx_fmat& k_space, ParametterSet& myparameter)
{
	_iteration_count = 0;
	float eps = 2.22044604925031 * (pow(10, -16));

	auto g0 = ComputeGradient(k_space, myparameter.wavelet_weight,
		myparameter.tv_weight, myparameter.typenorm, myparameter.mask, myparameter.undersampling_k_space);
	cx_fmat differential_recon_data = -g0;

	while (1)
	{
		auto recon_image = ifft2(k_space);
		auto differential_recon_image = ifft2(differential_recon_data);

		//prepare for calculate fidelity energy；
		auto recon_k_data = dot_multiply(k_space, myparameter.mask);
		auto differential_recon_k_data = dot_multiply(differential_recon_data, myparameter.mask);

		//prepare for calculate wavelet energy；
		auto recon_wavelet_data = Fw2DTransform(recon_image);
		auto differential_recon_wavelet_data = Fw2DTransform(differential_recon_image);

		//prepare for calculate tv energy；
		auto reconstruct_tv_data = TV2DTransform(recon_image);
		auto differential_reconstruct_tv_data = TV2DTransform(differential_recon_image);

		float initial_step = 0.0;
		float initial_energy = CalculateEnergy(recon_k_data,
			differential_recon_k_data,
			recon_wavelet_data,
			differential_recon_wavelet_data,
			reconstruct_tv_data,
			differential_reconstruct_tv_data,
			myparameter, initial_step);
		float step = myparameter.initial_line_search_step;
		float final_energy = CalculateEnergy(recon_k_data,
			differential_recon_k_data,
			recon_wavelet_data,
			differential_recon_wavelet_data,
			reconstruct_tv_data,
			differential_reconstruct_tv_data,
			myparameter, step);
		unsigned int line_search_times = 0;
		auto temp_g0 = conj_multiply(g0 ,differential_recon_data);
		auto sum_temp_g0 = accu(temp_g0);
		float energy_variation_g0 = abs(sum_temp_g0);

		while ((final_energy > (initial_energy - myparameter.line_search_alpha * step * energy_variation_g0)) &&
			(line_search_times < myparameter.max_line_search_times))
		{
			line_search_times = line_search_times + 1;
			step = step * myparameter.line_search_beta;
			final_energy = CalculateEnergy(recon_k_data,
				differential_recon_k_data,
				recon_wavelet_data,
				differential_recon_wavelet_data,
				reconstruct_tv_data,
				differential_reconstruct_tv_data,
				myparameter, step);
		}
		if (line_search_times == myparameter.max_line_search_times)
		{
			return cx_fmat();
		}
		if (line_search_times > 2)
		{
			myparameter.initial_line_search_step = myparameter.initial_line_search_step * myparameter.line_search_beta;
		}
		if (line_search_times < 1)
		{
			myparameter.initial_line_search_step = myparameter.initial_line_search_step / myparameter.line_search_beta;
		}

		k_space += differential_recon_data * step;

		auto g1 = ComputeGradient(k_space, myparameter.wavelet_weight, myparameter.tv_weight, 
			myparameter.typenorm, myparameter.mask, myparameter.undersampling_k_space);

		auto temp_g1 = square_module(g1);
		auto sum_energy_g1 = accu(temp_g1);
		//auto module_g0 = g0.module();								//
		auto power_module_g0 = square_module(g0);		//
		auto sum_energy_g0 = accu(power_module_g0);		//
		auto ellipse_factor = sum_energy_g1 / (sum_energy_g0 + eps);

		auto g0 = g1;
		differential_recon_data *= ellipse_factor;
		differential_recon_data += (-g1);
		++_iteration_count;
		auto temp_differential_reconstruct_data = module(differential_recon_data);
		auto differential_reconstruct_data_norm = accu(temp_differential_reconstruct_data);
		if ((_iteration_count > myparameter.max_conjugate_gradient_iteration_times) ||
			(differential_reconstruct_data_norm < myparameter.gradient_tollerance))
		{
			break;
		}
	}
	return k_space;
}

cx_fmat Yap::CompressedSensing::ComputeGradient(cx_fmat& in_data, float wavelet_weight, float tv_weight, float p_norm, 
	fmat mask, cx_fmat subsampled_kdata)
{
	auto gradient = GetFidelityTerm(in_data, mask, subsampled_kdata);
	auto image = ifft2(in_data);

	if (wavelet_weight)
	{
		auto wavelet_term = GetWaveletTerm(image, p_norm);
		wavelet_term *= wavelet_weight;
		gradient += wavelet_term;
	}

	if (tv_weight)
	{
		auto tv_term = GetTVTerm(image, p_norm);
		tv_term *= tv_weight;
		gradient += tv_term;
	}

	return gradient;
}

cx_fmat Yap::CompressedSensing::GetFidelityTerm(cx_fmat& in_data, fmat& mask, cx_fmat subsampled_kdata)
{
	return (dot_multiply(in_data, mask) - subsampled_kdata) * 2.0f;
}

cx_fmat Yap::CompressedSensing::GetWaveletTerm(cx_fmat& in_data, float p_norm)
{
	auto wavelet_data = Fw2DTransform(in_data);
	float epsilon = pow(10, -15);
	for (unsigned int i = 0; i < wavelet_data.n_rows; i++)
	{
	   for (unsigned int j = 0; j < wavelet_data.n_cols; j++)
		{
			float module_square = norm(wavelet_data(i, j));
			wavelet_data(i, j) *= static_cast<float>(pow(module_square + epsilon, p_norm / 2.0 - 1.0) * p_norm);
		}
	}
	return fft2(IFw2DTransform(wavelet_data));
}

cx_fmat Yap::CompressedSensing::GetTVTerm(cx_fmat& in_data, float p_norm)
{
	auto dx = TV2DTransform(in_data);
	float epsilon = static_cast<float>(pow(10, -15));

	for (unsigned int i = 0; i < 2; ++i)
	{
		cx_fmat& tv = dx.slice(i);
		for (unsigned int i = 0; i < tv.n_rows; ++i)
		{
			for (unsigned int j = 0; j < tv.n_cols; ++j)
			{
				float module_square = norm(tv(i, j));
				tv(i, j) *= static_cast<float>(pow(module_square + epsilon, p_norm / 2.0 - 1.0) * p_norm);
			}
		}		
	}
	return fft2(ITV2DTransform(dx));
}



unsigned int Yap::CompressedSensing::QuadLength(unsigned int length)
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

std::vector<float> Yap::CompressedSensing::Iconv(std::vector<float>& filter, std::vector<float>& row)
{
	unsigned int n = row.size();
	unsigned int p = filter.size();
	std::vector<float> xpadded(n + p);
	if (p <= n)
	{
		memcpy(xpadded.data(), row.data() + n - p, p * sizeof(float));
		memcpy(xpadded.data() + p, row.data(), n * sizeof(float));
	}
	else
	{
		std::vector<float> z(p);
		unsigned int imod;
		for (unsigned int i = 0; i < p; ++i)
		{
			imod = 1 + (p * n - p + i) - n * (unsigned)floor((p * n - p + i) / n);
			z[i] = row[imod];
		}
		memcpy(xpadded.data(), z.data(), p * sizeof(float));
		memcpy(xpadded.data() + p, row.data(), n * sizeof(float));
	}
	std::vector<float> ypadded(Filter(filter, 1, xpadded));
	std::vector<float> y(row.size());
	memcpy(y.data(), ypadded.data() + p, n * sizeof(float));
	return y;
}

std::vector<float> Yap::CompressedSensing::Filter(std::vector<float>& filter, unsigned int a, std::vector<float>& in_put)
{
	std::vector<float> temp_filter(filter);
	for (unsigned int i = 0; i < temp_filter.size(); ++i)
	{
		temp_filter.at(i) = temp_filter.at(i) / a;
	}
	std::vector<float> ypadded(in_put.size());
	for (unsigned int i = 0; i < in_put.size(); ++i)
	{
		if (i < temp_filter.size())
		{
			for (unsigned int j = 0; j <= i; ++j)
			{
				ypadded[i] += static_cast<float>(temp_filter.at(j) * in_put[i - j]);
			}
		}
		else
		{
			for (unsigned int j = 0; j < temp_filter.size(); ++j)
			{
				ypadded[i] += static_cast<float>(temp_filter.at(j) * in_put[i - j]);
			}
		}
	}
	return ypadded;
}

std::vector<float> Yap::CompressedSensing::MirrorFilt(std::vector<float>& filter)
{
	std::vector<float> mirror_filter(filter.size());
	for (unsigned int i = 0; i < filter.size(); ++i)
	{
		mirror_filter.at(i) = (-1) * (pow(-1, i + 1)) * filter.at(i);
	}
	return mirror_filter;
}

std::vector<float> Yap::CompressedSensing::Aconv(std::vector<float>& filter, std::vector<float>& row)
{
	auto n = row.size();
	auto p = filter.size();
	std::vector<float> xpadded(row);
	std::vector<float> temp_filter(filter);
	if (p < n)
	{
		xpadded.insert(xpadded.end(), xpadded.begin(), xpadded.begin() + p);
	}
	else
	{
		std::vector<float> z(p);
		unsigned int imod;
		for (unsigned int i = 0; i < p; ++i)
		{
			imod = 1 + i - n * (unsigned)floor(i / n);
			z[i] = row[imod];
		}
		xpadded.insert(xpadded.end(), z.begin(), z.end());
	}

	std::reverse(temp_filter.begin(), temp_filter.end());
	std::vector<float> ypadded(Filter(temp_filter, 1, xpadded));
	std::vector<float> y(row.size());
	memcpy(y.data(), ypadded.data() + p - 1, sizeof(float) * n);
	return y;
}

void Yap::CompressedSensing::SetFilterParams(unsigned int coarse_level)
{
	_coarse_level = coarse_level;
}

void Yap::CompressedSensing::GenerateFilter(wchar_t * filter_type_name, unsigned int coarse_level)
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

void Yap::CompressedSensing::SetFilter(float * filter, unsigned int size)
{
	_filter.resize(size);
	for (decltype(_filter.size()) i = 0; i < size; ++i)
	{
		_filter.at(i) = *filter;
		++filter;
	}
}

std::vector<float> Yap::CompressedSensing::DownSamplingLowPass(std::vector<float>& row, std::vector<float>& filter)
{
	std::vector<float> d1(Aconv(filter, row));
	auto n = d1.size();
	std::vector<float> d2;
	for (unsigned int i = 0; i < n - 1; i += 2)
	{
		d2.push_back(d1[i]);	//////////
	}
	return d2;
}

std::vector<float> Yap::CompressedSensing::DownSamplingHighPass(std::vector<float>& row, std::vector<float>& filter)
{
	//MirrorFilt
	std::vector<float> mirror_filter(filter.size());
	mirror_filter = MirrorFilt(filter);
	//Ishift
	std::vector<float> row_lshift(filter.size());
	row_lshift = LeftShift(row);
	std::vector<float> buffer(Iconv(mirror_filter, row_lshift));
	auto n = buffer.size();
	std::vector<float> d;
	for (unsigned int i = 0; i < n - 1; i += 2)
	{
		d.push_back(buffer[i]);
	}
	return d;
}

std::vector<float> Yap::CompressedSensing::LeftShift(std::vector<float>& row)
{
	std::vector<float> row_lshift(row.size());
	memcpy(row_lshift.data(), row.data() + 1, (row.size() - 1) * sizeof(float));
	*(row_lshift.end() - 1) = row.at(0);
	return row_lshift;
}

fmat Yap::CompressedSensing::DownSampling(fmat& output, std::vector<float>& filter, unsigned int nc)
{
	auto width = output.n_cols;
	float * bot = output.memptr();
	float * top = output.memptr() + nc / 2;
	for (unsigned int ix = 0; ix < nc; ++ix)//行变换
	{		
		std::vector<float> row(nc);//把一行所有的元素提取出来，低通放左边，高通放右边
		memcpy(row.data(), bot + ix * width, nc * sizeof(float));
		memcpy(bot + ix * width, DownSamplingLowPass(row, filter).data(), (nc / 2) * sizeof(float));
		memcpy(top + ix * width, DownSamplingHighPass(row, filter).data(), (nc / 2) * sizeof(float));
	}
	return output;
}


fmat Yap::CompressedSensing::IFWT2D(fmat& input, std::vector<float>& filter, unsigned int scale, ILongTaskListener * listener /*= nullptr*/)
{
	auto width = input.n_cols;
	auto height = input.n_rows;
	assert(width == height);

	fmat output(height, width);
	memcpy(output.memptr(), input.memptr(), width * height * sizeof(float));
	unsigned int J = QuadLength(width);//a power of 2
	unsigned int nc = static_cast<unsigned int>(pow(2, scale + 1));
	if (listener != nullptr)
	{
		listener->OnBegin();
		listener->OnProgress(0);
	}
	for (unsigned int jscal = 0; jscal < scale; ++jscal)    //  unsigned int jscal = scale; jscal <= J - 1; ++jscal
	{
		//列方向进行反小波变换
//		memcpy(&output(0, 0), &Transpose(output)(0, 0), width * height * sizeof(float));
		output = UpSampling(Transpose(output), filter, nc);
//		memcpy(&output(0, 0), &Transpose(output)(0, 0), width * height * sizeof(float));
		output = UpSampling(Transpose(output), filter, nc);
		nc = nc * 2;
		if (listener != nullptr)
		{
			listener->OnProgress((jscal + 1) * 100 / scale);
		}
	}
	if (listener != nullptr)
	{
		listener->OnEnd();
	}
	return output;
}

fmat Yap::CompressedSensing::UpSampling(fmat& output, std::vector<float>& filter, unsigned int nc)
{
	auto width = output.n_cols;
	float * top = output.memptr() + nc / 2;
	float * bot = output.memptr();
	for (unsigned int ix = 0; ix < nc; ++ix)				 // 行变换
	{
		std::vector<float> bot_row(nc / 2);
		memcpy(bot_row.data(), bot + ix * width, (nc / 2) * sizeof(float));
		std::vector<float> front(nc);
		memcpy(front.data(), UpSamplingLowPass(bot_row, filter).data(), nc * sizeof(float));
		std::vector<float> top_row(nc / 2);
		memcpy(top_row.data(), top + ix * width, (nc / 2) * sizeof(float));
		std::vector<float> back(nc);
		memcpy(back.data(), UpSamplingHighPass(top_row, filter).data(), nc * sizeof(float));
		std::vector<float> row(nc);
		for (unsigned int i = 0; i < nc; ++i)
		{
			row[i] = front[i] + back[i];
		}
		memcpy(bot, row.data(), nc * sizeof(float));
	}
	return output;
}

std::vector<float> Yap::CompressedSensing::UpSamplingLowPass(std::vector<float>& row, std::vector<float>& filter)
{
	std::vector<float> upsample_row(UpSampleInterpolateZero(row));
	std::vector<float> y;
	y = Iconv(filter, upsample_row);
	return y;
}

std::vector<float> Yap::CompressedSensing::UpSampleInterpolateZero(std::vector<float>& row)
{
	unsigned int s = 2;
	unsigned int n;
	n = row.size() * s;
	std::vector<float> y(n);
	y.assign(y.size(), 0);
	unsigned int j = 0;
	for (unsigned int i = 0; i < n; i += 2)
	{
		y[i] = row[j];
		++j;
	}
	return y;
}

std::vector<float> Yap::CompressedSensing::UpSamplingHighPass(std::vector<float>& row, std::vector<float>& filter)
{
	//MirrorFilt
	std::vector<float> mirror_filter(MirrorFilt(filter));
	//UpSampleN
	std::vector<float> upsample_row(UpSampleInterpolateZero(row));
	//Rshift
	std::vector<float> row_rshift(RightShift(upsample_row));
	std::vector<float> y(Aconv(mirror_filter, row_rshift));
	return y;
}

std::vector<float> Yap::CompressedSensing::RightShift(std::vector<float>& row)
{
	std::vector<float> row_lshift(row.size());
	*(row_lshift.data()) = *(row.end() - 1);
	memcpy(row_lshift.data() + 1, row.data(), (row.size() - 1) * sizeof(float));
	return row_lshift;
}

cx_fcube Yap::CompressedSensing::TV2DTransform(cx_fmat& in_data)
{
	unsigned int width = in_data.n_cols;
	unsigned int height = in_data.n_rows;

	cx_fcube tv_result(height, width, 2);
	tv_result.zeros();
	auto& vertical_tv = tv_result.slice(0);
	for (unsigned int i = 0; i < height - 1; ++i)
	{
		for (unsigned int j = 0; j < width; ++j)
		{
			vertical_tv(i, j) = in_data(i + 1, j) - in_data(i, j);
		}
	}

	auto& horizontal_tv = tv_result.slice(1);
	for (unsigned int i = 0; i < height; ++i)
	{
		for (unsigned int j = 0; j < width - 1; ++j)
		{
			horizontal_tv(i, j) = in_data(i, j + 1) - in_data(i, j);
		}
	}

	return tv_result;
}

cx_fmat Yap::CompressedSensing::ITV2DTransform(cx_fcube& in_data)
{
	unsigned int width = in_data.slice(0).n_cols;
	unsigned int height = in_data.slice(0).n_rows;

	cx_fmat itv_result(height, width);
	itv_result.zeros();
	cx_fmat temp_x(height, width);
	cx_fmat temp_y(height, width);

	auto& vertical_tv = in_data.slice(0);
	for (unsigned int j = 0; j < width; ++j)
	{
		for (unsigned int i = 0; i < height - 1; ++i)
		{
			if (i < 1)
				temp_x(i, j) = vertical_tv(i, j) * (-1.0f);
			else
				temp_x(i, j) = vertical_tv(i - 1, j) - vertical_tv(i, j);
		}
	}
	for (unsigned int j = 0; j < width; ++j)
	{
		temp_x(height - 1, j) = vertical_tv(height - 2, j);
	}

	auto& horizontal_tv = in_data.slice(1);
	for (unsigned int j = 0; j < width; ++j)
	{
		for (unsigned int i = 0; i < height; ++i)
		{
			if (j != 0)
			{
				temp_y(i, j) = horizontal_tv(i, j - 1) - horizontal_tv(i, j);
			}
			else
			{
				temp_y(i, j) = horizontal_tv(i, j) * (-1.0f);
				if (i != 0 )
				{
					temp_y(i - 1, width - 1) = horizontal_tv(i - 1, width - 2);
				}
			}
			temp_y(height - 1, width - 1) = horizontal_tv(height - 1, width - 2);
		}
	}
	itv_result = temp_x + temp_y;

	return itv_result;
}


// void Yap::CompressedSensing::FftShift(matrix<complex<float>>& data)
// {
// 	auto width = data.size2();
// 	auto height = data.size1();
// 	SwapBlock(&data(0, 0), &data(0, 0) + height / 2 * width + width / 2, width / 2, height / 2, width);
// 	SwapBlock(&data(0, 0) + width / 2, &data(0, 0) + height / 2 * width, width / 2, height / 2, width);
// }

// void Yap::CompressedSensing::SwapBlock(std::complex<float> * block1, std::complex<float> * block2, 
// 	size_t width, size_t height, size_t line_stride)
// {
// 	std::vector<std::complex<float>> swap_buffer;
// 	swap_buffer.resize(width);
// 
// 	auto cursor1 = block1;
// 	auto cursor2 = block2;
// 	for (size_t row = 0; row < height; ++row)
// 	{
// 		memcpy(swap_buffer.data(), cursor1, width * sizeof(std::complex<float>));
// 		memcpy(cursor1, cursor2, width * sizeof(std::complex<float>));
// 		memcpy(cursor2, swap_buffer.data(), width * sizeof(std::complex<float>));
// 
// 		cursor1 += line_stride;
// 		cursor2 += line_stride;
// 	}
// }

fmat Yap::CompressedSensing::FWT2D(fmat& input, std::vector<float>& filter, unsigned int scale, ILongTaskListener * listener)
{
	unsigned int width = input.n_cols;
	unsigned int height = input.n_rows;
	assert(width == height);
	fmat output(height, width);
	memcpy(output.memptr(), input.memptr(), width * height * sizeof(float));

	unsigned int J = QuadLength(width);//a _iwavelet_slice_index of 2
	unsigned int nc = width;

	assert(J >= scale);
	if (listener != nullptr)
	{
		listener->OnBegin();
		listener->OnProgress(0);
	}
	for (unsigned int jscal = 0; jscal < scale; ++jscal)     // unsigned int jscal = J - 1; jscal >= scale; --jscal
	{
		output = DownSampling(output, filter, nc);// 行方向下采样
//		memcpy(&output(0, 0), &(Transpose(output)(0, 0)), width * height * sizeof(float));
		output = DownSampling(Transpose(output), filter, nc);//列方向下采样
//		memcpy(&output(0, 0), &(Transpose(output)(0, 0)), width * height * sizeof(float));
		output = Transpose(output);
		nc = nc / 2;
		if (listener != nullptr)
		{
			listener->OnProgress((jscal + 1) * 100 / scale);
		}
	}
	if (listener != nullptr)
	{
		listener->OnEnd();
	}
	return output;
}

cx_fmat Yap::CompressedSensing::Fw2DTransform(cx_fmat& input, ILongTaskListener * listener)
{
//	int t = sizeof(int);	//
	cx_fmat& in_data(input);
	unsigned int width = in_data.n_cols;
	unsigned int height = in_data.n_rows;
	fmat in_real_part(height, width);
	fmat in_imaginary_part(height, width);
	cx_fmat wave_data(height, width);
	//提取实部、虚部
	in_real_part = real(in_data);
	in_imaginary_part = imag(in_data);
	
	//CGenerateFilter filter(_T("db"), 3, 4); //第二个参数决定基的类型，第三个参数决定小波级数
	GenerateFilter(L"daub2.flt", 4);
	//小波变换
	auto output_real_part = FWT2D(in_real_part, _filter, _coarse_level, listener);
	auto output_imaginary_part = FWT2D(in_imaginary_part, _filter, _coarse_level, listener);
	//组合复数
	for (unsigned int i = 0; i < height; ++i)
	{
		for (unsigned int j = 0; j < width; ++j)
		{
			wave_data(i, j) = std::complex<float>(output_real_part(i, j), output_imaginary_part(i, j));
		}
	}

	return wave_data;
}

cx_fmat Yap::CompressedSensing::IFw2DTransform(cx_fmat& input, ILongTaskListener * listener /*= nullptr*/)
{
	cx_fmat& in_data(input);
	auto width = in_data.n_cols;
	auto height = in_data.n_rows;
	fmat in_real_part(height, width);
	fmat in_imaginary_part(height, width);
	cx_fmat iwave_data(height, width);

	//提取实部、虚部
	in_real_part = real(in_data);
	in_imaginary_part = imag(in_data);

	GenerateFilter(L"daub2.flt", 4);
	//反小波变换
	auto output_real_part = IFWT2D(in_real_part, _filter, _coarse_level, listener);
	auto output_imaginary_part = IFWT2D(in_imaginary_part, _filter, _coarse_level, listener);
	//组合复数
	for (size_t i = 0; i < height; ++i)
	{
		for (size_t j = 0; j < width; ++j)
		{
			iwave_data(i, j) = std::complex<float>(output_real_part(i, j), output_imaginary_part(i, j));
		}
	}
	return iwave_data;
}

float Yap::CompressedSensing::CalculateEnergy(cx_fmat& recon_k_data, 
	                                          cx_fmat& differential_recon_k_data, 
	                                          cx_fmat& recon_wavelet_data, 
	                                          cx_fmat& differential_recon_wavelet_data, 
	                                          cx_fcube& recon_tv_data, 
	                                          cx_fcube& differential_recon_tv_data, 
	                                          ParametterSet& myparameter, 
	                                          float step_length)
{
	float total_energy = 0.0;
	float fidelity_energy = 0.0;
	cx_fmat temp_fidelity_energy = recon_k_data + differential_recon_k_data * step_length - myparameter.undersampling_k_space;

	auto power_module_temp_fidelity_energy = square_module(temp_fidelity_energy);
	fidelity_energy = accu(power_module_temp_fidelity_energy);
	float epsilon = static_cast<float>(pow(10, -15));
//	matrix<float> epsilon_m(recon_wavelet_data.size1(), recon_wavelet_data.size2());

	float wavelet_energy = 0.0;
	if (myparameter.wavelet_weight)
	{
		cx_fmat temp_wavelet_energy = recon_wavelet_data + differential_recon_wavelet_data * step_length;
		//auto module_wavelet_energy = temp_wavelet_energy.module();
		fmat power_module_temp_wavelet_energy = square_module(temp_wavelet_energy) + epsilon;  //模的平方
		//auto temp = DataType::power(power_module_temp_wavelet_energy, p_norm/2);
		//数学上等同于上式：p_norm = 1；
		auto temp = sqrt_root(power_module_temp_wavelet_energy);
		wavelet_energy = accu(temp) * myparameter.wavelet_weight;
	}

	float tv_energy = 0.0;
	if (myparameter.tv_weight)
	{
		for (unsigned int i = 0; i < 2; ++i)
		{
			cx_fmat temp_tv_energy = recon_tv_data.slice(i) + differential_recon_tv_data[i] * step_length;
			//auto module_tv_energy = temp_tv_energy.module();
			fmat power_module_temp_tv_energy = square_module(temp_tv_energy) + epsilon;			//模的平方
		    //auto temp = DataType::power(power_module_temp_tv_energy, p_norm/2);
			//数学意义上等同于上式：p_norm = 1
			auto temp = sqrt_root(power_module_temp_tv_energy);
			tv_energy += accu(temp) * myparameter.tv_weight;
		}
	}

	total_energy = fidelity_energy + wavelet_energy + tv_energy;
	return total_energy;
}


fmat Yap::CompressedSensing::Transpose(fmat& in_data)
{
	fmat output(in_data.n_cols, in_data.n_rows);

	for (unsigned int i = 0; i < in_data.n_rows; ++i)
	{
		for (unsigned int j = 0; j < in_data.n_cols; ++j)
		{
			output(j, i) = in_data(i, j);
		}
	}
	return output;
}

arma::cx_fmat Yap::CompressedSensing::Transpose(arma::cx_fmat & in_data)
{
	cx_fmat output(in_data.n_cols, in_data.n_rows);

	for (unsigned int i = 0; i < in_data.n_rows; ++i)
	{
		for (unsigned int j = 0; j < in_data.n_cols; ++j)
		{
			output(j, i) = in_data(i, j);
		}
	}
	return output;
}

fmat Yap::CompressedSensing::square_module(cx_fmat& input)
{
	fmat normal(input.n_rows, input.n_cols);
	for (unsigned int i = 0; i < input.n_rows; ++i)
	{
		for (unsigned int j = 0; j < input.n_cols; ++j)
		{
			normal(i, j) = norm(input(i, j));
		}
	}
	return normal;
}

// matrix<float> Yap::CompressedSensing::fill(float value, matrix<float>& input)
// {
// 	for (unsigned int i = 0; i < input.size1(); ++i)
// 	{
// 		for (unsigned int j = 0; j < input.size2(); ++j)
// 		{
// 			input(i, j) = value;
// 		}
// 	}
// 	return input;
// }

fmat Yap::CompressedSensing::sqrt_root(fmat& input)
{
	fmat output(input.n_rows, input.n_cols);
	for (unsigned int i = 0; i < input.n_rows; ++i)
	{
		for (unsigned int j = 0; j < input.n_cols; ++j)
		{
			output(i, j) = sqrt(input(i, j));
		}
	}
	return output;
}


cx_fmat Yap::CompressedSensing::conj_multiply(cx_fmat& input_1, cx_fmat& input_2)
{
	cx_fmat result(input_1.n_rows, input_1.n_cols);
	for (unsigned int i = 0; i < input_1.n_rows; ++i)
	{
		for (unsigned int j = 0; j < input_1.n_cols; ++j)
		{
			result(i, j) = conj(input_1(i, j)) * input_2(i, j);
		}
	}
	return result;
}

fmat Yap::CompressedSensing::module(cx_fmat& input)
{
	fmat result(input.n_rows, input.n_cols);
	for (unsigned int i = 0; i < input.n_rows; ++i)
	{
		for (unsigned int j = 0; j < input.n_cols; ++j)
		{
			result(i, j) = abs(input(i, j));
		}
	}
	return result;
}

arma::cx_fmat Yap::CompressedSensing::dot_multiply(arma::cx_fmat in_data, arma::fmat mask)
{
	cx_fmat out_data(in_data.n_rows, in_data.n_cols);
	for (unsigned int i = 0; i < in_data.n_rows; ++i)
	{
		for (unsigned int j = 0; j < in_data.n_cols; ++j)
		{
			out_data(i, j) = in_data(i, j) * mask(i, j);
		}
	}
	return out_data;
}

// void Yap::CompressedSensing::Plan(size_t width, size_t height, bool inverse, bool inplace)
// {
// 	std::vector<fftwf_complex> data(width * height);
// 
// 	if (inplace)
// 	{
// 		_fft_plan = fftwf_plan_dft_2d(int(width), int(height), (fftwf_complex*)data.data(),
// 			(fftwf_complex*)data.data(),
// 			inverse ? FFTW_BACKWARD : FFTW_FORWARD,
// 			FFTW_MEASURE);
// 	}
// 	else
// 	{
// 		std::vector<fftwf_complex> result(width * height);
// 		_fft_plan = fftwf_plan_dft_2d(int(width), int(height), (fftwf_complex*)data.data(),
// 			(fftwf_complex*)result.data(),
// 			inverse ? FFTW_BACKWARD : FFTW_FORWARD,
// 			FFTW_MEASURE);
// 	}
// 	_plan_data_width = static_cast<unsigned int> (width);
// 	_plan_data_height = static_cast<unsigned int> (height);
// 	_plan_inverse = inverse;
// 	_plan_in_place = inplace;
// }



