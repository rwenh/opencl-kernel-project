#include "opencl_wrapper/platform.hpp"
#include <iostream>

namespace platform {
    std::vector<cl_platform_id> get_platforms() {
        cl_uint num_platforms = 0;
        clGetPlatformIDs(0, nullptr, &num_platforms);

        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

        return platforms;
    }

    std::vector<cl_device_id> get_devices(cl_platform_id platform, cl_device_type type) {
        cl_uint num_devices = 0;
        clGetDeviceIDs(platform, type, 0, nullptr, &num_devices);

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, type, num_devices, devices.data(), nullptr);

        return devices;
    }

    std::string get_platform_name(cl_platform_id platform) {
        char buffer[1024];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, nullptr);
        return std::string(buffer);
    }

    std::string get_device_name(cl_device_id device) {
        char buffer[1024];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
        return std::string(buffer);
    }

    void print_platform_info(cl_platform_id platform) {
        char buffer[1024];

        std::cout << "=== Platform Info ===" << std::endl;

        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, nullptr);
        std::cout << "Name: " << buffer << std::endl;

        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(buffer), buffer, nullptr);
        std::cout << "Vendor: " << buffer << std::endl;

        clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, nullptr);
        std::cout << "Version: " << buffer << std::endl;
    }

    void print_device_info(cl_device_id device) {
        char buffer[1024];
        cl_uint compute_units;
        cl_ulong global_mem;

        std::cout << "=== Device Info ===" << std::endl;

        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
        std::cout << "Name: " << buffer << std::endl;

        clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, nullptr);
        std::cout << "Vendor: " << buffer << std::endl;

        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
        std::cout << "Compute Units: " << compute_units << std::endl;

        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);
        std::cout << "Global Memory: " << (global_mem / 1024 / 1024) << " MB" << std::endl;
    }
}