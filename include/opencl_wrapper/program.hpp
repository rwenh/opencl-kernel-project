#pragma once

#include<CL/opencl.h>
#include<string.h>
#include<vector>
#include<unordered_map>
#include<stdexcept>
#include<fstream>
#include<sstream>

namespace program {
// --Error helper--------------
inline void check(cl_int err, const char* msg){
	if(err !=CL_SUCCESS)
		throw std::runtime_error(std::string(msg)+" (  code"+std::to_string(err)+ ") ");
}
// ---Build log-----------------
inline std::string get_build_log(cl_program program, cl_device_id device){
	size_t size=0;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,&size);
	std::string log(size, '\0');
	clGetProgramBuildInfo(program, device,CL_PROGRAM_BUILD_LOG, size, log.data(),nullptr);
	if(! log.empty() && log.back()=='\0')log.pop_back();
	return log;
}
//---Build helper(shared by source/binary/IL paths)---------------------------
inline void build_program(cl_program program,
		const std::vector<cl_device_id>& devices={},
		const std::string& options =""){
	cl_uint n = static_cast<cl_uint>(devices.size());
	const cl_device_id* ptr = devices.empty()?nullptr:devices.data();
	cl_int err =clBuildProgram(program,n,ptr,options.c_str(),nullptr,nullptr);
	if(err != CL_SUCCESS){
		std::string log;
		if (devices.empty()){
			for(auto d:devices)log+=get_build_log(program,d)+"\n";
		}else{
			//fetch devices from program
			cl_uint count =0;
			clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(count),&count,nullptr);
			std::vector<cl_device_id> devs(count);
			clGetProgramInfo(program, CL_PROGRAM_DEVICES, count*sizeof(cl_device_id),devs.data(),nullptr);
			for(auto d:devs)log +=get_build_log(program,d)+"\n";
		}
		throw std::runtime_error("clBuildProgram failed(code " + std::to_string(err) + "):\n"+log);
	}
}
//-------------------------------------------RAII program---------------------------------------------
struct Program{
	cl_program handle=nullptr;
	Program()=default;
	//From source string
	static Program  from_source(cl_context context, const std::string& source, const std::vector<cl_device_id>& devices={ },const std::string& options=""){
		cl_int err=CL_SUCCESS;
		const char* src=source.c_str();
		size_t len=source.size();
		cl_program p= clCreateProgramWithSource(context,1,&src,&len,&err);
		Program prog; prog.handle=p;
		build_program(p,devices,options);
		return prog;
	}
	// From source file
	static Program from_file(cl_context context, const std::string& path, const std::vector<cl_device_id>& devices={},const std::string& options=""){
		std::ifstream f(path);
		if(!f)throw std::runtime_error("Cannot open kernel file "+path);
		std::ostringstream ss; ss<< f.rdbuf();
		return from_source(context,ss.str(), devices, options);
	}
	//From SPIR-V IL(opencl 2.1+ /3.0)
	static Program from_il(cl_context context, const std::vector<uint8_t>& il, const std::vector<cl_device_id>& devices={}, const std::string& options=""){
		cl_int err =CL_SUCCESS;
		cl_program p=clCreateProgramWithIL(context,il.data(),il.size(),&err);
		check(err,"clCreateProgramWithIL");
		Program prog; prog.handle=p;
		build_program(p,devices,options);
		return prog;
	}
	// From compiled binary
	static Program from_binary(cl_context context,
			const std::vector<cl_device_id>& devices,
			const std::vector<std::vector<uint8_t>>& binaries,
			const std::string& options=""){
		if(devices.size()!=binaries.size())
			throw std::invalid_argument("Device and binary counts must match");
		std::vector<const uint8_t*>ptrs;
		std::vector<size_t>sizes;
		for (auto& b :binaries) { ptrs.push_back(b.data()); sizes.push_back(b.size()) ; }
		cl_int err=CL_SUCCESS;
		std::vector<cl_int>bin_status(devices.size());
		cl_program p = clCreateProgramWithBinary(context,
				static_cast<cl_uint>(devices.size()),
				devices.data(),sizes.data(),ptrs.data(),
				bin_status.data(),&err);
		check(err,"clCreateProgramWithBinary");
		for(size_t i=0; i<bin_status.size();++i)
			if(bin_status[i]!=CL_SUCCESS)
				throw std::runtime_error("Binary invalid for device " + std::to_string(i));
		Program prog;prog.handle=p;
		build_program(p,devices,options);
		return prog;
	}
	~Program(){if (handle)clReleaseProgram(handle);}
	Program (const Program&)=delete;
	Program& operator= (const Program&)=delete;
	Program(Program&& o) noexcept:handle(o.handle){o.handle=nullptr;}
	Program& operator=(Program&& o)noexcept{
		if(this!=&o){if(handle)clReleaseProgram(handle);handle=o.handle;o.handle=nullptr;}
		return *this;
	}
	operator cl_program()const {return handle;}
	bool valid() const {return handle!= nullptr;}
	//Extract compiled binary for caching
	std::vector<uint8_t>get_binary(cl_device_id device)const{
		cl_uint num_devs=0;
		clGetProgramInfo(handle, CL_PROGRAM_NUM_DEVICES,sizeof(num_devs),&num_devs,nullptr);
		std::vector<cl_device_id>devs(num_devs);
		clGetProgramInfo(handle, CL_PROGRAM_DEVICES,num_devs*sizeof(cl_device_id),devs.data(),nullptr);
		std::vector<size_t>sizes(num_devs);
		clGetProgramInfo(handle, CL_PROGRAM_BINARY_SIZES,num_devs*sizeof(size_t),sizes.data(),nullptr);
		std::vector<std::vector<uint8_t>> bins(num_devs);
		std::vector<uint8_t*>ptrs(num_devs);
		for(cl_uint i=0; i<num_devs;++i){bins[i].resize(sizes[i]);ptrs[i]=bins[i].data();}
		clGetProgramInfo(handle,CL_PROGRAM_BINARIES,num_devs*sizeof(uint8_t*),ptrs.data(),nullptr);
		for(cl_uint i=0;i<num_devs;++i)
			if(devs[i]==device)return bins[i];
		throw std::runtime_error ("Device not found in program");
	}
};
//-----RAII kernel------------------------------------------------------------------
struct Kernel{
	cl_kernel handle=nullptr;
	std::string name;
	Kernel()=default;
	Kernel(cl_program program, const std::string& kernel_name)
	:name(kernel_name){
		cl_int err=CL_SUCCESS;
		handle=clCreateKernel(program, kernel_name.c_str(), &err);
		check(err,("clCreateKernel: " + kernel_name).c_str());
	}
	~Kernel(){if(handle)clReleaseKernel(handle);}
	Kernel(const Kernel&)=delete;
	Kernel& operator=(const Kernel&)=delete;
	Kernel(Kernel&& o)noexcept:handle(o.handle),name(std::move(o.name)){o.handle=nullptr;}
	Kernel& operator=(Kernel&& o)noexcept{
		if(this!=&o){if(handle)clReleaseKernel(handle);handle=o.handle;name=std::move(o.name);o.handle=nullptr;}
		return *this;
	}
	operator cl_kernel() const{return handle;}
	bool valid()const{return handle!=nullptr;}
	//Set a kernel argument by index
	template<typename T>
	void set_arg(cl_uint index, const T& value){
		check(clSetKernelArg(handle,index,sizeof(T),&value),
				("clSetKernelArg[" +std::to_string(index)+ "]").c_str());
	}
	//Specialisation for cl_mem(buffers)
	void set_arg(cl_uint index, cl_mem buffer){
		check(clSetKernelArg(handle,index,sizeof(cl_mem), &buffer),
				("clSetKernelArg["+ std::to_string(index)+"]cl_mem").c_str());
	}
	//Local Memory Placeholder
	void set_local_arg(cl_uint index, size_t bytes){
		check(clSetKernelArg(handle,index,bytes,nullptr),
				("clSetkernelArg["+std::to_string(index)+"] local").c_str());
	}
};
//-----Create all kernels in a program-------------------------------------------------------
inline std::unordered_map<std::string, Kernel>create_all_kernels(cl_program program){
	cl_uint count=0;
	check(clCreateKernelsInProgram(program,0,nullptr,&count),"clCreateKernelsInProgram count");
	std::vector<cl_kernel>raw(count);
	check(clCreateKernelsInProgram(program,count,raw.data(),nullptr),"clCreateKernelIsInProgram");
	std::unordered_map<std::string,Kernel>result;
	for(auto k:raw){
		size_t size=0;
		clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 0,nullptr,&size);
		std::string kname(size,'\0');
		clGetKernelInfo(k,CL_KERNEL_FUNCTION_NAME,size,kname.data(),nullptr);
		if(!kname.empty()&&kname.back()=='\0')kname.pop_back();
		Kernel kobj; kobj.handle=k;kobj.name=kname;
		result.emplace(kname,std::move(kobj));
	}
	return result;
}
}// namespace program
