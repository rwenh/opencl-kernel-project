#pragma once

#include <CL/opencl.h>
#include <iostream>
#include <string>
#include <vector>

namespace platform {

void check(cl_int err, const char *msg);

std::vector<cl_platform_id> get_platforms();
std::vector<cl_device_id> get_devices(cl_platform_id platform,
                                       cl_device_type type = CL_DEVICE_TYPE_ALL);
std::vector<cl_device_id> get_all_devices(cl_device_type type = CL_DEVICE_TYPE_ALL);
cl_device_id select_best_device(cl_device_type preferred = CL_DEVICE_TYPE_GPU);

std::string get_platform_info_str(cl_platform_id platform, cl_platform_info param);
std::string get_device_info_str(cl_device_id device, cl_device_info param);
std::string get_platform_name(cl_platform_id platform);
std::string get_device_name(cl_device_id device);

template <typename T>
inline T get_device_info(cl_device_id device, cl_device_info param) {
    T value{};
    check(clGetDeviceInfo(device, param, sizeof(T), &value, nullptr),
          "clGetDeviceInfo");
    return value;
}

cl_uint get_compute_units(cl_device_id d);
cl_ulong get_global_mem(cl_device_id d);
cl_ulong get_local_mem(cl_device_id d);
size_t get_max_work_group_size(cl_device_id d);
bool supports_fp64(cl_device_id d);

void print_platform_info(cl_platform_id platform, std::ostream &out = std::cout);
void print_device_info(cl_device_id device, std::ostream &out = std::cout);

} // namespace platform
