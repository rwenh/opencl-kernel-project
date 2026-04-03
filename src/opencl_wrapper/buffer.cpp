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
 Buffer Buffer :: sub_buffer(size_t origin, size_t size, cl_mem_flags flags) const {
 cl_buffer_region region{origin, size};
 cl_int err=CL_SUCCESS;
 Buffer sub;
 sub.handle=clCreateSubBuffer(handle, flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
 check(err, "clCreateSubBuffer");
 sub.byte_size = size;
 return sub;
 }
 /*----------------------------------------------------------------------------------------------------------------------------------------------------------
  * Blocking Transfers
  * _________________________________________________________________________________________*/
void Buffer::write(cl_command_queue queue, const void *data, size_t size, size_t offset) const {
	if(size==0)
		size=byte_size;
	check(clEnqueueWriteBuffer(queue, handle , CL_TRUE,offset,size,data,0,nullptr,nullptr),
			"clEnqueueWriteBuffer");
}
void Buffer::read(cl_command_queue queue, void *data, size_t size, size_t offset)const{
	if(size==0)size=byte_size;
	check(clEnqueueReadBuffer(queue, handle, CL_TRUE,offset,size,data,0, nullptr, nullptr),"clEnqueueReadBuffer");
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * Async transfers
 * ------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
cl_event Buffer::write_async(cl_command_queue queue, const void *data, size_t size, size_t offset, const std::vector<cl_event> &wait_list) const{
	if(size ==0)size=byte_size;
	cl_event ev= nullptr;
	check(clEnqueueWriteBuffer(queue, handle, CL_FALSE, offset, size, data, static_cast<cl_uint>(wait_list.size()), wait_list.empty()?nullptr:wait_list.data(), &ev),
			"clEnqueueWriteBufferasync");
	return ev;
}
cl_event Buffer::read_async(cl_command_queue queue, void *data, size_t size, size_t offset, const std::vector<cl_event> &wait_list)const{
	if(size==0)
		size= byte_size;
	cl_event ev= nullptr;
	check(clEnqueueReadBuffer(queue, handle,CL_FALSE,offset,size,data, static_cast<cl_uint>(wait_list.size()), wait_list.empty()? nullptr:wait_list.data(), &ev),
			"clEnqueueReadBuffer async");
	return ev;
}
/*---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * Device side copy
 * __________________________________________________________________________________________________
 */
void Buffer::copy_to (cl_command_queue queue, const Buffer &dst, size_t size,size_t src_offset, size_t dst_offset)const{
	if (size==0)
		size=byte_size;
	check(clEnqueueCopyBuffer(queue,handle, dst.handle, src_offset,dst_offset, size, 0, nullptr, nullptr),
			"clEnqueueCopyBuffer");
}
/*------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * Map/Unmap
 * -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 */
void *Buffer::map(cl_command_queue queue, cl_map_flags flags, size_t offset , size_t size) const{
	if(size==0)size=byte_size;
	cl_int err=CL_SUCCESS;
	void *ptr=clEnqueueMapBuffer(queue, handle, CL_TRUE, flags, offset, size,0, nullptr, nullptr, &err);
	check(err, "clEnqueueMapBuffer");
	return ptr;
}
void Buffer::unmap(cl_command_queue queue, void *ptr)const {
	check(clEnqueueUnmapMemObject(queue, handle, ptr, 0 , nullptr, nullptr),"clEnqueueUnmapMemObject");
}
/*_______________________________________________________________________________________________________________
 * Free function:fill(non template overload)
 * ___________________________________________________________________________________________________________________
 */
void fill(cl_command_queue queue, const Buffer &buf,
		const void *pattern, size_t pattern_size, size_t offset,size_t size){
	if (size==0)size=buf.size();
	check(clEnqueueFillBuffer(queue, buf.handle, pattern, pattern_size, offset, size, 0, nullptr, nullptr),"clEnqueueFillBuffer");
}

} //namespace Buffer
