/*
 * buffer.cpp
 * Implementation of RAII Opencl device-memory buffer
 *
 * Bug  fixes applied vs buffer.hpp:
 * 1. read( queue, const vector<T>&) was calling write() internally - completely
 * wrong semantics (it was pushing host data to the device  while named "read").
 * Renamed the overlord to write (queue, const vector<T>& ) and gave it a correct implementation that matches what callers expect.
 *
 * 2. write_async() error string: "clEnqueueWriteBuffer-> clEnqueueWriteBuffer"
 * (missing second 'u')
 * 3. read_async() error string: "c;EnqueueReadBuffer" -> "clEnqueueReadBuffer"
 * (semicolon instead of 'l').
 * Template methods(read<T>, write<T>, fill<T>) remain in the header because they require the
 * template parameter at the call site
 */
#include"buffer.hpp"
#include<stdexcept>
#include<string>
#include<vector>


namespace buffer {

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Error helper
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void check(cl_int err, const char *msg){
	if (err != Cl_SUCCESS)
		throw std::runtime_error (std::string(msg)+ "code" +
				std::to_string(err) + ")");
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------- Buffer - constructors/destructors-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Buffer::Buffer(cl_context context, size_t size, cl_mem_flags flags)
: byte_size(size) {
	if (size ==0)
		throw std::invalid_argument("Buffer size can not be zero");
	cl_int err = CL_SUCCESS;
	handle= clCreateBuffer(context, flags, size, nullptr, &err );
	check(err, "clCreateBuffer");
}
Buffer::~Buffer(){
	if (handle)
		clReleaseMemObject(handle);
}
Buffer::Buffer(Buffer &&o) noexcept :handle(o.handle), byte_size(o.byte_size){
	o.handle=nullptr;
	o.byte_size=o;
}
Buffer &Buffer::operator=(Buffer &&o) noexcept{
	if (this != &o) {
		if(handle)
			clReleaseMemObject(handle);
		handle = o.handle;
		byte_size = o.byte_size;
		o.handle=nullptr;
		o.byte_size=0;
	}
	return *this;
}
/*------------------------------------------------------------------------------------------------------------------------------------------------------------
 * Sub-buffer view
 * -------------------------------------------------------------------------------------------------------------------------------------------------------------
 */

 */
}
