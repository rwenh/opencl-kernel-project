#include "opencl_wrapper/platform.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <string_view>

namespace platform {
    void check_cl_error(cl_int error, const std::string& message) {
        using namespace std::string_literals;
        if (error != CL_SUCCESS) {
            throw std::runtime_error(message + " (Error code: "s + std::to_string(error) + ")");
        }
    }

    [[nodiscard]] std::vector<cl_platform_id> get_platforms() {
        cl_uint num_platforms = 0;

        cl_int error = clGetPlatformIDs(0, nullptr, &num_platforms);
        check_cl_error(error, "Failed to get number of platforms");

        if (num_platforms == 0) {
            return {};
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        error = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        check_cl_error(error, "Failed to get platform IDs");

        return platforms;
    }

    [[nodiscard]] std::vector<cl_device_id> get_devices(cl_platform_id platform, cl_device_type type) {
        if (platform == nullptr) {
            throw std::invalid_argument("Platform ID cannot be null");
        }

        cl_uint num_devices = 0;

        cl_int error = clGetDeviceIDs(platform, type, 0, nullptr, &num_devices);
        if (error == CL_DEVICE_NOT_FOUND) {
            return {};
        }
        check_cl_error(error, "Failed to get number of devices");

        if (num_devices == 0) {
            return {};
        }

        std::vector<cl_device_id> devices(num_devices);
        error = clGetDeviceIDs(platform, type, num_devices, devices.data(), nullptr);
        check_cl_error(error, "Failed to get device IDs");

        return devices;
    }

    [[nodiscard]] std::string get_platform_name(cl_platform_id platform) {
        if (platform == nullptr) {
            throw std::invalid_argument("Platform ID cannot be null");
        }

        std::array<char, 1024> buffer{};
        size_t returned_size = 0;

        cl_int error = clGetPlatformInfo(platform, CL_PLATFORM_NAME,
                                         buffer.size(), buffer.data(), &returned_size);
        check_cl_error(error, "Failed to get platform name");

        return std::string(buffer.data(), returned_size);
    }

    [[nodiscard]] std::string get_device_name(cl_device_id device) {
        if (device == nullptr) {
            throw std::invalid_argument("Device ID cannot be null");
        }

        std::array<char, 1024> buffer{};
        size_t returned_size = 0;

        cl_int error = clGetDeviceInfo(device, CL_DEVICE_NAME,
                                       buffer.size(), buffer.data(), &returned_size);
        check_cl_error(error, "Failed to get device name");

        return std::string(buffer.data(), returned_size);
    }

    void print_platform_info(cl_platform_id platform) {
        if (platform == nullptr) {
            throw std::invalid_argument("Platform ID cannot be null");
        }

        std::array<char, 1024> buffer{};

        std::cout << "=== Platform Info ===" << std::endl;

        auto get_platform_info = [&](cl_platform_info param, std::string_view label) {
            cl_int error = clGetPlatformInfo(platform, param, buffer.size(), buffer.data(), nullptr);
            check_cl_error(error, std::string("Failed to get platform ") + std::string(label));
            std::cout << label << ": " << buffer.data() << std::endl;
        };

        get_platform_info(CL_PLATFORM_NAME, "Name");
        get_platform_info(CL_PLATFORM_VENDOR, "Vendor");
        get_platform_info(CL_PLATFORM_VERSION, "Version");
    }

    void print_device_info(cl_device_id device) {
        if (device == nullptr) {
            throw std::invalid_argument("Device ID cannot be null");
        }

        std::array<char, 1024> buffer{};
        cl_uint compute_units = 0;
        cl_ulong global_mem = 0;

        std::cout << "=== Device Info ===" << std::endl;

        auto get_device_info_str = [&](cl_device_info param, std::string_view label) {
            cl_int error = clGetDeviceInfo(device, param, buffer.size(), buffer.data(), nullptr);
            check_cl_error(error, std::string("Failed to get device ") + std::string(label));
            std::cout << label << ": " << buffer.data() << std::endl;
        };

        auto get_device_info_scalar = [&](auto param, auto& value, std::string_view label) {
            cl_int error = clGetDeviceInfo(device, param, sizeof(value), &value, nullptr);
            check_cl_error(error, std::string("Failed to get device ") + std::string(label));
        };

        get_device_info_str(CL_DEVICE_NAME, "Name");
        get_device_info_str(CL_DEVICE_VENDOR, "Vendor");

        get_device_info_scalar(CL_DEVICE_MAX_COMPUTE_UNITS, compute_units, "compute units");
        std::cout << "Compute Units: " << compute_units << std::endl;

        get_device_info_scalar(CL_DEVICE_GLOBAL_MEM_SIZE, global_mem, "global memory size");
        std::cout << "Global Memory: " << (global_mem / 1024 / 1024) << " MB" << std::endl;
    }
}