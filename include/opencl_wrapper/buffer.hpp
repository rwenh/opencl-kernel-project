#pragma once

#include <CL/opencl.h>
#include <vector>

namespace buffer {
    cl_mem create_buffer(cl_context context, size_t size,
                         cl_mem_flags flags = CL_MEM_READ_WRITE);

    void write_buffer(cl_command_queue queue, cl_mem buffer,
                      const void* data, size_t size);

    void read_buffer(cl_command_queue queue, cl_mem buffer,
                     void* data, size_t size);

    template<typename T>
    cl_mem create_buffer(cl_context context, const std::vector<T>& data,
                         cl_mem_flags flags = CL_MEM_READ_WRITE) {

    }

    void release_buffer(cl_mem buffer);
}