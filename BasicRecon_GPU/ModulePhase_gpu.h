#pragma once
#include "Interface\Implement\ProcessorImpl.h"
#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>

namespace Yap
{
	class ModulePhase_gpu :
		public ProcessorImpl
	{
	public:
		ModulePhase_gpu();
		virtual ~ModulePhase_gpu();

		virtual IProcessor * Clone() override;

		virtual bool Input(const wchar_t * name, IData * data) override;

	};
}
