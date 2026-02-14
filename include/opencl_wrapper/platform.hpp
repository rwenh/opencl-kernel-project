#pragma once

#include <CL/opencl.h>
#include <string>
#include <vector>

namespace platform {
    std::vector<cl_platform_id> get_platforms();

    std::vector<cl_device_id> get_devices(cl_platform_id platform,
                                          cl_device_type type = CL_DEVICE_TYPE_GPU);

    std::string get_platform_name(cl_platform_id platform);
    std::string get_device_name(cl_device_id device);
    void print_platform_info(cl_platform_id platform);
    void print_device_info(cl_device_id device);
}