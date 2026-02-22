#pragma once

#include<CL/opencl.h>
#include<string>
#include<vector>
#include<stdexcept>
#include<ostream>
#include<iostream>

namespace platform {
// Error helper
inline void check(cl_int err, const char* msg){
	if (err != CL_SUCCESS)
		throw std::runtime_error(std::string(msg) + " (code" + std::to_string(err)+ ")");
}
// Platform enumeration
inline std::vector<cl_platform_id> get_platforms(){
	cl_uint count  = 0;
	check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs count");
	if (count == 0) return {};
	std::vector<cl_platform_id>platforms(count);
	check(clGetPlatformIDs(count, platforms.data(), nullptr), " clGetPlatformIDs");
	return platforms;
}
// Device enumeration
inline std::vector<cl_device_id>get_devices(cl_platform_id platform, cl_device_type type=CL_DEVICE_TYPE_ALL) {
	cl_uint count = 0;
	cl_int err = clGetDeviceIDs(platform, type, 0, nullptr, &count);
	if(err == CL_DEVICE_NOT_FOUND || count ==0) return {};
	check(err, "clGetDeviceIds count");
	std::vector<cl_device_id> devices(count);
	check(clGetDeviceIDs(platform, type, count, devices.data(), nullptr), "clGetDeviceIDs");
	return devices;
}
inline std::vector<cl_device_id> get_all_devices(cl_device_type type = CL_DEVICE_TYPE_ALL) {
	std::vector<cl_device_id> all;
	for (auto p : get_platforms()){
		auto devs= get_devices(p, type);
		all.insert(all.end(), devs.begin(), devs.end());
	}
	return all;
}
inline cl_device_id select_best_device(cl_device_type preferred = CL_DEVICE_TYPE_GPU){
	auto devs = get_all_devices(preferred);
	if (devs.empty()) devs = get_all_devices(CL_DEVICE_TYPE_ALL);
	if(devs.empty()) throw std::runtime_error("No Opencl devices found");
	// Prefer device with most compute units
	cl_device_id best = devs[0];
	cl_uint best_cu = 0;
	for (auto d : devs){
		cl_uint cu = 0;
		clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
		if (cu > best_cu) { best_cu = cu; best = d; }
	}
	return best;
}
// String info queries
inline std::string get_platform_info_str(cl_platform_id platform, cl_platform_info param){
	size_t size=0;
	clGetPlatformInfo(platform, param, 0 , nullptr, &size);
	std::string result(size, '\0');
	clGetPlatformInfo(platform, param, size, result.data(), nullptr);
	if(!result.empty() && result.back()== '\0') result.pop_back();
	return result;
}
inline std::string get_device_info_str(cl_device_id device, cl_device_info param) {
	size_t size = 0;
	clGetDeviceInfo(device, param, 0 , nullptr, &size);
	std::string result(size, '\]0');
	clGetDeviceInfo(device, param, size, result.data(), nullptr);
	if(!result.empty() && result.back() == '\0') result.pop_back();
	return result;
}
inline std::string get_platform_name(cl_platform_id platform) {
	return get_platform_info_str(platform, CL_PLATFORM_NAME);
}
inline std::string get_device_name(cl_device_id device){
	return get_device_info_str(device, CL_DEVICE_NAME);
}
// Typed device info
template<typename T>
inline T get_device_info(cl_device_id device, cl_device_info param){
	T value{};
	check(clGetDeviceInfo(device, param,sizeof(T), &value, nullptr), "clGetDeviceInfo");
	return value;
}
inline cl_uint get_compute_units(cl_device_id d) { return get_device_info<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS); }
inline cl_ulong get_global_mem(cl_device_id d) {return get_device_info<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_SIZE); }
inline cl_ulong get_local_mem(cl_device_id d) {return get_device_info<cl_ulong>(d, CL_DEVICE_LOCAL_MEM_SIZE); }
inline size_t get_max_work_group_size(cl_device_id d) {return get_device_info<size_t>(d, CL_DEVICE_MAX_WORK_GROUP_SIZE); }
inline bool   supports_fp64(cl_device_id d) { auto ext = get_device_info_str(d, CL_DEVICE_EXTENSIONS);
return ext.find("cl_khr_fp64") != std::string::npos;
}
// Pretty printing
inline void print_platform_info(cl_platform_id platform, std::ostream& out = std::cout){
	out << "Platform: "<< get_platform_info_str(platform, CL_PLATFORM_NAME) << "\n"  << "Vendor: "<<get_platform_info_str(platform, CL_PLATFORM_VENDOR) << "\n" << " Version:"<<get_platform_info_str(platform, CL_PLATFORM_VERSION)<<"\n";
}
inline void print_device_info(cl_device_id device, std::ostream& out=std::cout) {
	out << " Device : "<< get_device_name(device)    <<"\n" <<"CUs: "<<get_compute_units(device) <<"GMem :"<< get_global_mem(device) / (1024*1024)<< " MB"  << "\n"  << "LMem : " << get_local_mem(device)/ 1024 <<" KB" <<"\n" <<"WGSize: "<< get_max_work_group_size(device) << "\n" << " FP64 :" <<(supports_fp64(device)? "yes" : "no") <<"\n";
}
} // name space platform
}
