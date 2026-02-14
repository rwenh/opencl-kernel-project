#pragma once

#include <CL/opencl.h>
#include <vector>

namespace context {
    cl_context create_context(const std::vector<cl_device_id>& devices);
    cl_context create_context(cl_device_id device);

    cl_command_queue create_command_queue(cl_context context, cl_device_id device);

    void release_context(cl_context context);
    void release_command_queue(cl_command_queue queue);
}