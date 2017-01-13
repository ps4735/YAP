#include "stdafx.h"
#include "CompressedSensing_cuda.h"
#include "Interface/Client/DataHelper.h"
#include "Interface/Implement/DataObject.h"
#include "Recon_Runtime.h"
#include "ReconBase.h"
#include "vector"

using namespace std;
using namespace Yap;

CompressedSensing_cuda::CompressedSensing_cuda():
	ProcessorImpl(L"CompressedSensing_cuda"),
	_elapsedTime(0.0f)
{
	AddInput(L"Input", YAP_ANY_DIMENSION, DataTypeComplexFloat);
	AddInput(L"Mask", 2, DataTypeFloat);
	AddOutput(L"Output", 2, DataTypeComplexFloat);
}


CompressedSensing_cuda::~CompressedSensing_cuda()
{
}

IProcessor * Yap::CompressedSensing_cuda::Clone()
{
	return new (nothrow) CompressedSensing_cuda(*this);
}

bool Yap::CompressedSensing_cuda::Input(const wchar_t * port, IData * data)
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
		unsigned int width = input_data.GetWidth();
		unsigned int height = input_data.GetHeight();
		std::vector<float> mask(width * height);
		std::vector<std::complex<float>> raw_data(width * height);

		memcpy(mask.data(), GetDataArray<float>(_mask.get()), width * height * sizeof(float));
		memcpy(raw_data.data(), GetDataArray<complex<float>>(data), width * height * sizeof(complex<float>));
		ParameterSet myparameter = { 200, float(pow(10,-30)), 8, 1, 0.002f, 0.005f, 0.01f, 0.8f, 2, mask, raw_data };

		std::complex<float> * recon_data = nullptr;
		try
		{
			recon_data = new std::complex<float>[width * height];
		}
		catch (std::bad_alloc&)
		{
			return false;
		}
		Recon_Runtime recon;
		recon.GetReconData(mask, raw_data, myparameter, width, height, _elapsedTime, recon_data);
		_elapsedTime = _elapsedTime / 1000.0;
		Dimensions dimensions;
		dimensions(DimensionReadout, 0U, input_data.GetWidth())
			(DimensionPhaseEncoding, 0U, input_data.GetHeight());
		auto output = YapShared(new ComplexFloatData(recon_data, dimensions, nullptr, true));

		Feed(L"Output", output.get());
	}
	else
	{
		return false;
	}
	return true;
}
