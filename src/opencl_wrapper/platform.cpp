#include "opencl_wrapper/platform.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

namespace platform {
    void check_cl_error(cl_int error, const std::string& message) {
        if (error != CL_SUCCESS) {
            throw std::runtime_error(message + " (Error code: " + std::to_string(error) + ")");
        }
    }

    std::vector<cl_platform_id> get_platforms() {
        cl_int error;
        cl_uint num_platforms = 0;

        error = clGetPlatformIDs(0, nullptr, &num_platforms);
        check_cl_error(error, "Failed to get number of platforms");

        if (num_platforms == 0) {
            return std::vector<cl_platform_id>();
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        error = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        check_cl_error(error, "Failed to get platform IDs");

        return platforms;
    }

    std::vector<cl_device_id> get_devices(cl_platform_id platform, cl_device_type type) {
        if (platform == nullptr) {
            throw std::invalid_argument("Platform ID cannot be null");
        }

        cl_int error;
        cl_uint num_devices = 0;

        error = clGetDeviceIDs(platform, type, 0, nullptr, &num_devices);
        if (error == CL_DEVICE_NOT_FOUND) {
            return std::vector<cl_device_id>();
        }
        check_cl_error(error, "Failed to get number of devices");

        if (num_devices == 0) {
            return std::vector<cl_device_id>();
        }

        // Allocate vector and get device IDs
        std::vector<cl_device_id> devices(num_devices);
        error = clGetDeviceIDs(platform, type, num_devices, devices.data(), nullptr);
        check_cl_error(error, "Failed to get device IDs");

        return devices;
    }

    std::string get_platform_name(cl_platform_id platform) {
        if (platform == nullptr) {
            throw std::invalid_argument("Platform ID cannot be null");
        }

        char buffer[1024] = {0};
        size_t returned_size = 0;

        cl_int error = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, &returned_size);
        check_cl_error(error, "Failed to get platform name");

        return std::string(buffer, returned_size > 0 ? returned_size - 1 : 0);
    }

    std::string get_device_name(cl_device_id device) {
        if (device == nullptr) {
            throw std::invalid_argument("Device ID cannot be null");
        }

        char buffer[1024] = {0};
        size_t returned_size = 0;

        cl_int error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, &returned_size);
        check_cl_error(error, "Failed to get device name");

        return std::string(buffer, returned_size > 0 ? returned_size - 1 : 0);
    }

    void print_platform_info(cl_platform_id platform) {
        if (platform == nullptr) {
            throw std::invalid_argument("Platform ID cannot be null");
        }

        cl_int error;
        char buffer[1024] = {0};

        std::cout << "=== Platform Info ===" << std::endl;

        error = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, nullptr);
        check_cl_error(error, "Failed to get platform name");
        std::cout << "Name: " << buffer << std::endl;

        error = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(buffer), buffer, nullptr);
        check_cl_error(error, "Failed to get platform vendor");
        std::cout << "Vendor: " << buffer << std::endl;

        error = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, nullptr);
        check_cl_error(error, "Failed to get platform version");
        std::cout << "Version: " << buffer << std::endl;
    }

    void print_device_info(cl_device_id device) {
        if (device == nullptr) {
            throw std::invalid_argument("Device ID cannot be null");
        }

        cl_int error;
        char buffer[1024] = {0};
        cl_uint compute_units = 0;
        cl_ulong global_mem = 0;

        std::cout << "=== Device Info ===" << std::endl;

        error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
        check_cl_error(error, "Failed to get device name");
        std::cout << "Name: " << buffer << std::endl;

        error = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, nullptr);
        check_cl_error(error, "Failed to get device vendor");
        std::cout << "Vendor: " << buffer << std::endl;

        error = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
        check_cl_error(error, "Failed to get compute units");
        std::cout << "Compute Units: " << compute_units << std::endl;

        error = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, nullptr);
        check_cl_error(error, "Failed to get global memory size");
        std::cout << "Global Memory: " << (global_mem / 1024 / 1024) << " MB" << std::endl;
    }
}