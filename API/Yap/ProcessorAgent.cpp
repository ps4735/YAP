#include "ProcessorAgent.h"
#include "Interface\IProperties.h"

#include <cassert>

using namespace Yap;

CProcessorAgent::CProcessorAgent(IProcessor* processor) :
	_processor(processor)
{
}

CProcessorAgent::~CProcessorAgent()
{
}

Yap::IProcessor * Yap::CProcessorAgent::Clone()
{
	assert(_processor != nullptr);
	return _processor->Clone();
}

void Yap::CProcessorAgent::DeleteThis()
{
	if (_processor != nullptr)
	{
		_processor->DeleteThis();
		_processor = nullptr;
	}
}

const wchar_t * Yap::CProcessorAgent::GetClassId()
{
	assert(_processor != nullptr);
	return _processor->GetClassId();
}

void Yap::CProcessorAgent::SetClassId(const wchar_t * id)
{
	assert(_processor != nullptr);
	_processor->SetClassId(id);
}

const wchar_t * Yap::CProcessorAgent::GetInstanceId()
{
	assert(_processor != nullptr);
	return _processor->GetInstanceId();
}

void Yap::CProcessorAgent::SetInstanceId(const wchar_t * instance_id)
{
	assert(_processor != nullptr);
	_processor->SetInstanceId(instance_id);
}

Yap::IPortContainer * Yap::CProcessorAgent::GetInputPorts()
{
	assert(_processor != nullptr);
	return _processor->GetInputPorts();
}

Yap::IPortContainer * Yap::CProcessorAgent::GetOutputPorts()
{
	assert(_processor != nullptr);
	return _processor->GetOutputPorts();
}

Yap::IPropertyContainer * Yap::CProcessorAgent::GetProperties()
{
	assert(_processor != nullptr);
	return _processor->GetProperties();
}

bool Yap::CProcessorAgent::LinkProperty(const wchar_t * property_id, const wchar_t * param_id)
{
	assert(_processor != nullptr);
	return _processor->LinkProperty(property_id, param_id);
}

bool Yap::CProcessorAgent::UpdateProperties(IPropertyContainer * params)
{
	assert(_processor != nullptr);
	return _processor->UpdateProperties(params);
}

bool Yap::CProcessorAgent::Link(const wchar_t * output, IProcessor * next, const wchar_t * next_input)
{
	assert(_processor != nullptr);
	return _processor->Link(output, next, next_input);
}

bool Yap::CProcessorAgent::Input(const wchar_t * name, IData * data)
{
	assert(_processor != nullptr);
	return _processor->Input(name, data);
}

bool Yap::CProcessorAgent::SetInt(const wchar_t * property_name,
	int value)
{
	assert(_processor != nullptr);
	auto properties = _processor->GetProperties();
	assert(properties != nullptr);

	IProperty * property = properties->GetProperty(property_name);
	assert(property != nullptr && property->GetType() == PropertyInt);

	IInt * int_property = dynamic_cast<IInt*>(property);
	assert(int_property != nullptr);

	int_property->SetInt(value);

	return true;
}

bool Yap::CProcessorAgent::SetBool(const wchar_t * property_name, bool value)
{
	assert(_processor != nullptr);
	auto properties = _processor->GetProperties();
	assert(properties != nullptr);

	IProperty * property = properties->GetProperty(property_name);
	assert(property != nullptr && property->GetType() == PropertyBool);
	IBoolean * bool_property = dynamic_cast<IBoolean*>(property);
	assert(bool_property != nullptr);

	bool_property->SetBool(value);

	return true;
}

bool Yap::CProcessorAgent::SetFloat(const wchar_t * property_name, double value)
{
	assert(_processor != nullptr);
	auto properties = _processor->GetProperties();
	assert(properties != nullptr);

	IProperty * property = properties->GetProperty(property_name);
	assert(property != nullptr && property->GetType() == PropertyFloat);

	IDouble * float_property = dynamic_cast<IDouble*>(property);
	assert(float_property != nullptr);

	float_property->SetDouble(value);

	return true;
}

bool Yap::CProcessorAgent::SetString(const wchar_t* property_name, 
	const wchar_t* value)
{
	assert(_processor != nullptr);
	auto properties = _processor->GetProperties();
	assert(properties != nullptr);

	IProperty * property = properties->GetProperty(property_name);
	assert(property != nullptr && property->GetType() == PropertyString);

	IString * string_property = dynamic_cast<IString*>(property);
	assert(string_property != nullptr);

	string_property->SetString(value);

	return true;
}

Yap::CProcessorAgent::operator bool()
{
	return _processor != nullptr;
}
