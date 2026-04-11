/*
 * context.cpp
 * Implementation of RAII opencl context and command queue parsers
 *
 * Build: part of opencl_wrapper library target
 */
#include"context.hpp"
#include<stdexcept>
#include<string>
#include<vector>

namespace context{
/*-----------------------------------------------------------------------------------------------------------------------------
 * Error helper
 * -----------------------------------------------------------------------------------------------------------------------------
 */
void check(cl_int err, const char *msg){
	if(err != CL_SUCCESS)
		throw std::runtime_error(std::string(msg) + "(code " +
				std::to_string(err)+ " ) ");
}
/*-----------------------------------------------------------------------------------------------------------------------------
 * contexts-- contructor/destructor
 * -----------------------------------------------------------------------------------------------------------------------------
 */
context::context(const std::vector<cl_device_id> &devices,
		cl_platform_id platform,
		void(CL_CALLBACK *pfn_notify) (const char *, const void * , size_t, void *),
		void * USER_DATA){
	cl_int err= CL_SUCCESS;
	std::vector<cl_context_properties> props;
	if(platform){
		props = {CL_CONTEXT_PLATFORM,
				reinterpret_cast<cl_context_properties>(platform), 0};
	}
	handle= clCreateContext(props.empty()? nullptr: props.data(),
			static_cast<cl_uint>(devices.size()),
			devices.data(), pfn_notify, USER_DATA,&err);
	check(err, "clCreateContext");
	}
context::context (cl_device_id device, cl_platform_id platform)
: context(std:: vector<cl_device_id>{device}, platform) {}
context::~context() {
	if(handle)
		clReleaseContext(handle);
}
context::context (context &&o) noexcept:handle(o.handle){
	o.handle=nullptr;
}
context &context::operator =(context &&o) noexcept{
	if(this != &o){
		if(handle)
			clReleaseContext(handle);
		handle=o.handle;
		o.handle= nullptr;
	}
	return *this;
}
/*---------------------------------------------------------------------------------------------------------------------------------------
 * Queue constructor deconstructor
 * --------------------------------------------------------------------------------------------------------------------------------------
 */
QUEUE::QUEUE (cl_context ctx, cl_device_id device,
		cl_command_queue_properties properties) {
	cl_int err= CL_SUCCESS;
	if(properties != 0){
		//Only build the properties array when there are flags to set.
		//Passing a non-null empty array is technically valid but wasteful
		const cl_queue_properties props[]= {CL_QUEUE_PROPERTIES, properties, 0};
		handle= clCreateCommandQueueWithProperties(ctx, device, props, &err);
	}else{
		handle=clCreateCommandQueueWithProperties(ctx, device, nullptr, &err);
	}
	check(err, "clCreateCommandQueueWithProperties");
}
QUEUE::~QUEUE(){
	if(handle){
		clFlush(handle);
		clFinish(handle);
		clReleaseCommandQueue(handle);
	}
}
QUEUE::QUEUE(QUEUE &&o) noexcept: handle(o.handle){
	o.handle= nullptr;
}
QUEUE &QUEUE::operator= (QUEUE &&o) noexcept{
	if(this != &o){
		if(handle){
			clFlush(handle);
			clFinish(handle);
			clReleaseCommandQueue(handle);
		}
		handle=o.handle;
		o.handle=nullptr;
	}
	return *this;
}
void QUEUE::flush() const {clFlush(handle);}
void QUEUE::finish() const {clFinish(handle);}
}
