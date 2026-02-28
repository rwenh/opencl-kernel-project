#pragma once
#include<CL/opencl.h>
#include<vector>
#include<stdexcept>
#include<string>
#include<cstring>
namespace buffer {
//Error helper----
inline void check(cl_int err, const char* msg){
	if (err !=  CL_SUCCESS )
		throw std::runtime_error (std::string(msg) + " ( code " + std::to_string(err) + ") ");
}
//RAII Buffer
struct Buffer {
	cl_mem handle = nullptr;
	size_t byte_size=0;

	Buffer() = default;
	//Raw size allocation
	Buffer (cl_context context, size_t size,
			cl_mem_flags flags= CL_MEM_READ_WRITE)
	:  byte_size(size){
		if (size == 0) throw std::invalid_argument("Buffer size can not be zero");
		cl_int err =CL_SUCCESS;
		handle=clCreateBuffer(context, flags, size, nullptr,&err);
		check(err, "clCreateBuffer");
	}
	//From host vector- copies data to device
	template<typename T>
	Buffer(cl_context context, const std::vector<T>& data,
			cl_mem_flags flags= CL_MEM_READ_WRITE)
			: byte_size (data.size() * sizeof (T)){
		if(data.empty()) throw std::invalid_argument("Buffer data cannot be empty ");
		cl_int err= CL_SUCCESS;
		handle=clCreateBuffer(context,  flags| CL_MEM_COPY_HOST_PTR,
				byte_size,
				//clCreateBuffer takes void* , not const void*
				const_cast<T*>(data.data()), &err);
		check(err, "clCreateBuffer (from vector)");
	}
	//Sub-buffer view
	Buffer sub_buffer (size_t origin, size_t size,
			cl_mem_flags flags= CL_MEM_READ_WRITE) const{
		cl_buffer_region region {origin, size};
		cl_int err= CL_SUCCESS;
		Buffer sub;
		sub.handle = clCreateSubBuffer(handle, flags,
				CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
		check(err, "clCreateSubBuffer");
		sub.byte_size= size;
		return sub;
	}
	~Buffer() { if (handle) clReleaseMemObject(handle); }
	Buffer(const Buffer&)=delete;
	Buffer& operator= (const Buffer&)= delete;

	Buffer(Buffer&& o) noexcept:handle (o.handle), byte_size(o.byte_size){
		o.handle=nullptr; o.byte_size=0;
	}
	Buffer& operator=(Buffer&& o) noexcept{
		if(this !=&0){
			if (handle) clReleaseMemObject(handle);
			handle=o.handle; byte_size= o.byte_size;
			o.handle= nullptr; o.byte_size=0;
		}
		return *this;
	}
	operator cl_mem() const{return handle;}
	bool valid() const{return handle != nullptr;}
	size_t size() const{return byte_size;}
	//Blocking transfers
	void write(cl_command_queue queue, const void* data, size_t size=0,
			size_t offset=0)const{
		if(size==0)size=byte_size;
		check(clEnqueueWriteBuffer(queue, handle, CL_TRUE, offset, size,
				data,0,nullptr,nullptr),
				"clEnqueueWriteBuffer");
	}
	void read(cl_command_queue queue ,void* data, size_t size= 0 ,
			size_t offset=0)const{
		if(size==0)size=byte_size;
		check(clEnqueueReadBuffer(queue, handle, CL_TRUE, offset, size,
				data, 0, nullptr, nullptr),
				"clEnqueueReadBuffer");
	}
	template<typename T>
	void read(cl_command_queue queue, const std::vector<T>& data, size_t offset=0)const{
		write(queue, data.data(), data.size() * sizeof(T), offset);
	}
	template<typename T>
	void read(cl_command_queue queue, std::vector<T>& data, size_t offset=0)const{
		data.resize(byte_size/sizeof(T));
		read(queue, data.data(), byte_size, offset);
	}
	template<typename T>
	std::vector<T> read(cl_command_queue queue)const{
		std::vector<T> result;
		read< T>(queue, result);
		return result;
	}
	//Async transfers(Returns event)
	cl_event write_async(cl_command_queue queue, const void* data, size_t size=0,
			size_t offset=0,
			const std::vector<cl_event>& wait_list ={})const{
		if(size==0)size=byte_size;
		cl_event ev=nullptr;
		check(clEnqueueWriteBuffer(queue, handle, CL_FALSE,offset, size, data,
				static_cast<cl_uint>(wait_list.size()),
				wait_list.empty()?nullptr:wait_list.data(), &ev),
				"clEnqueWriteBuffer async");
		return ev;
	}
	cl_event read_async (cl_command_queue queue, void* data, size_t size=0,
			size_t offset=0,
			const std::vector<cl_event>& wait_list ={} )const{
		if (size==0) size=byte_size;
		cl_event ev= nullptr;
		check(clEnqueueReadBuffer(queue, handle, CL_FALSE, offset, size, data,
				static_cast<cl_uint>(wait_list.size()),
				wait_list.empty() ? nullptr:wait_list.data(), &ev),
				"c;EnqueueReadBuffer async");
		return ev;
	}
	//Device - side copy
	void copy_to(cl_command_queue queue, const Buffer& dst,
			size_t size=0, size_t src_offset=0, size_t dst_offset=0)const{
		if(size==0)size=byte_size;
		check(clEnqueueCopyBuffer(queue, handle, dst.handle,
				src_offset, dst_offset, size,
				0, nullptr, nullptr),
				"clEnqueueCopyBuffer");
	}
	//Map/unmap--
	void* map(cl_command_queue queue,
			cl_map_flags flags=CL_MAP_READ| CL_MAP_WRITE,
			size_t offset=0, size_t size=0)const{
		if(size==0)size=byte_size;
		cl_int err= CL_SUCCESS;
		void* ptr=clEnqueueMapBuffer(queue, handle,CL_TRUE,flags,
				offset,size, 0 , nullptr,nullptr,&err);
		check(err, "clEnqueueMapBuffer");
		return ptr;
	}
	void unmap(cl_command_queue queue,void* ptr ) const{
		check(clEnqueueUnmapMemObject(queue, handle, ptr, 0, nullptr, nullptr),
				"clEnqueueUnmapMemObject");
	}
};
//Free functions(convenience)
inline void fill(cl_command_queue queue, const Buffer& buf,
		const void* pattern, size_t pattern_size,
		size_t offset=0 , size_t size=0){
	if (size==0)size=buf.size();
	check(clEnqueueFillBuffer(queue, buf.handle, pattern, pattern_size,
			offset,size, 0, nullptr, nullptr),
			"clEnqueueFillBuffer");
}
template<typename T>
inline void fill(cl_command_queue queue, const Buffer& buf,const T& value){
	fill(queue,buf, &value, sizeof(T));
}
}//namespace buffer
