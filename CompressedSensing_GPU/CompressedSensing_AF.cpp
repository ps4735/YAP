#include "stdafx.h"
#include "CompressedSensing_AF.h"
#include "Interface/Client/DataHelper.h"
#include "Interface/Implement/DataObject.h"
#include "CompressedSensing_gpu.h"

using namespace std;
using namespace Yap;
CompressedSensing_AF::CompressedSensing_AF():
	ProcessorImpl(L"CompressedSensing_AF")
{
	AddInput(L"Input", YAP_ANY_DIMENSION, DataTypeComplexFloat);
	AddInput(L"Mask", 2, DataTypeFloat);
	AddOutput(L"Output", 2, DataTypeComplexFloat);
}


CompressedSensing_AF::~CompressedSensing_AF()
{
}

IProcessor * Yap::CompressedSensing_AF::Clone()
{
	return new (nothrow) CompressedSensing_AF(*this);
}

bool Yap::CompressedSensing_AF::Input(const wchar_t * port, IData * data)
{
	if (std::wstring(port) == L"Mask")
	{
		_mask = YapShared(data);
	}
	else if (std::wstring(port) == L"Input")
	{
		if (!_mask)
			return false;

		DataHelper input_data(data);
		DataHelper input_mask(_mask.get());
		auto width = input_data.GetWidth();
		auto height = input_data.GetHeight();

		af::array mask(input_data.GetHeight(), input_data.GetWidth(), (float*)GetDataArray<float>(_mask.get()), afHost);
		af::array undersampled_data(input_data.GetHeight(), input_data.GetWidth(), (af::cfloat*)GetDataArray<std::complex<float>>(data), afHost);

		ParameterSet myparameter = { 200, float(pow(10,-30)), 8, 1, 0.002f, 0.005f, 0.01f, 0.8f, 2, mask, undersampled_data };

		CompressedSensing_gpu CS;
		auto recon_data = CS.Reconstruct(undersampled_data, myparameter);

		std::complex<float> * recon = nullptr;
		try
		{
			recon = new std::complex<float>[input_data.GetWidth() * input_data.GetHeight()];
		}
		catch (std::bad_alloc&)
		{
			return false;
		}

		recon_data.host(recon);
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
