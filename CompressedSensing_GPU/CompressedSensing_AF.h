#pragma once
#include "Interface\Implement\ProcessorImpl.h"
#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

namespace Yap
{
	class CompressedSensing_AF :
		public ProcessorImpl
	{
	public:
		CompressedSensing_AF();
		virtual ~CompressedSensing_AF();

		virtual IProcessor * Clone() override;

		virtual bool Input(const wchar_t * name, IData * data) override;

	private:
		SmartPtr<IData> _mask;
	};
}
