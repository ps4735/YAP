#include "stdafx.h"
#include "ModulePhase_gpu.h"
#include "Interface/Client/DataHelper.h"
#include <complex>


using namespace af;
using namespace Yap;


ModulePhase_gpu::ModulePhase_gpu():
	ProcessorImpl(L"ModulePhase_gpu")
{
	AddInput(L"Input", YAP_ANY_DIMENSION, DataTypeComplexDouble | DataTypeComplexFloat);
	AddOutput(L"Module", YAP_ANY_DIMENSION, DataTypeDouble | DataTypeFloat);
	AddOutput(L"Phase", YAP_ANY_DIMENSION, DataTypeDouble | DataTypeFloat);
}


ModulePhase_gpu::~ModulePhase_gpu()
{
}

IProcessor * Yap::ModulePhase_gpu::Clone()
{
	return new (std::nothrow) ModulePhase_gpu(*this);
}

bool Yap::ModulePhase_gpu::Input(const wchar_t * port, IData * data)
{
	if (std::wstring(port) != L"Input")
		return false;

	if (data->GetDataType() != DataTypeComplexDouble && data->GetDataType() != DataTypeComplexFloat)
		return false;

	DataHelper input_data(data);

	auto want_module = OutportLinked(L"Module");
	auto want_phase = OutportLinked(L"Phase");

	if (want_module)
	{
		if (data->GetDataType() == DataTypeComplexDouble)
		{
			auto module = YapShared(new DoubleData(data->GetDimensions()));
			array input_array(input_data.GetDataSize(), (cdouble*)GetDataArray<std::complex<double>>(data), afHost);
			array module_array = abs(input_array);
			try
			{
				module_array.host(GetDataArray<double>(module.get()));
			}
			catch (af::exception& e)
			{
				fprintf(stderr, "%s\n", e.what());
			}
			
			
			Feed(L"Module", module.get());
		}
		else
		{
			auto module = YapShared(new FloatData(data->GetDimensions()));
			array input_array(input_data.GetDataSize(), (cfloat*)GetDataArray<std::complex<float>>(data), afHost);
			array module_array = abs(input_array);
			try
			{
				module_array.host(GetDataArray<float>(module.get()));
			}
			catch (af::exception& e)
			{
				fprintf(stderr, "%s\n", e.what());
			}

			Feed(L"Module", module.get());
		}
	}

	if (want_phase)
	{
		if (data->GetDataType() == DataTypeComplexDouble)
		{
			auto phase = YapShared(new DoubleData(data->GetDimensions()));
			array input_array(input_data.GetDataSize(), (cdouble*)GetDataArray<std::complex<double>>(data), afHost);
			array phase_array = arg(input_array);
			try
			{
				phase_array.host(GetDataArray<double>(phase.get()));
			}
			catch (af::exception& e)
			{
				fprintf(stderr, "%s\n", e.what());
			}

			Feed(L"Phase", phase.get());
		}
		else
		{
			auto phase = YapShared(new FloatData(data->GetDimensions()));
			array input_array(input_data.GetDataSize(), (cfloat*)GetDataArray<std::complex<float>>(data), afHost);
			array phase_array = arg(input_array);
			try
			{
				phase_array.host(GetDataArray<float>(phase.get()));
			}
			catch (af::exception& e)
			{
				fprintf(stderr, "%s\n", e.what());
			}

			Feed(L"Phase", phase.get());
		}
	}
	return true;
}
