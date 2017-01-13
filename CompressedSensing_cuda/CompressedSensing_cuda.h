#pragma once
#include "Interface/Implement/ProcessorImpl.h"


namespace Yap
{
	class CompressedSensing_cuda :
		public ProcessorImpl
	{
	public:
		CompressedSensing_cuda();
		virtual ~CompressedSensing_cuda();

		virtual IProcessor * Clone() override;

		virtual bool Input(const wchar_t * name, IData * data) override;

	protected:
		SmartPtr<IData> _mask;
		float _elapsedTime;

	};
}
