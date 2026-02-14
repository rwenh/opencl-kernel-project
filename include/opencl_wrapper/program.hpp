#pragma once

#include <CL/opencl.h>
#include <string>
#include <vector>

namespace program {
    cl_program create_from_source(cl_context context, const std::string& source);

    void build_program(cl_program program,
                       const std::vector<cl_device_id>& devices,
                       const std::string& options = "");

    std::string get_build_log(cl_program program, cl_device_id device);

    cl_kernel create_kernel(cl_program program, const std::string& kernel_name);

    void release_program(cl_program program);
    void release_kernel(cl_kernel kernel);
}