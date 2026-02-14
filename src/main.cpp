#define CL_TARGET_OPENCL_VERSION 300

#include "opencl_wrapper/platform.hpp"
#include <iostream>
#include <exception>

int main() {
    try {
        std::cout << "=== OpenCL Platform and Device Information ===\n\n";

        auto platforms = platform::get_platforms();

        if (platforms.empty()) {
            std::cout << "No OpenCL platforms found on this system.\n";
            std::cout << "Please ensure OpenCL drivers are installed.\n";
            return 1;
        }

        std::cout << "Found " << platforms.size() << " platform(s)\n";
        std::cout << "==========================================\n\n";

        for (size_t i = 0; i < platforms.size(); ++i) {
            try {
                std::cout << "Platform #" << (i + 1) << ":\n";
                platform::print_platform_info(platforms[i]);

                std::cout << "\n--- Device Information ---\n";

                auto all_devices = platform::get_devices(platforms[i], CL_DEVICE_TYPE_ALL);

                if (all_devices.empty()) {
                    std::cout << "  No devices found for this platform.\n";
                } else {
                    std::cout << "  Found " << all_devices.size() << " device(s)\n\n";

                    auto cpu_devices = platform::get_devices(platforms[i], CL_DEVICE_TYPE_CPU);
                    auto gpu_devices = platform::get_devices(platforms[i], CL_DEVICE_TYPE_GPU);
                    auto accelerator_devices = platform::get_devices(platforms[i], CL_DEVICE_TYPE_ACCELERATOR);

                    if (!cpu_devices.empty()) {
                        std::cout << "  CPU Devices: " << cpu_devices.size() << "\n";
                    }
                    if (!gpu_devices.empty()) {
                        std::cout << "  GPU Devices: " << gpu_devices.size() << "\n";
                    }
                    if (!accelerator_devices.empty()) {
                        std::cout << "  Accelerator Devices: " << accelerator_devices.size() << "\n";
                    }

                    std::cout << "\n  Detailed device information:\n";

                    for (size_t j = 0; j < all_devices.size(); ++j) {
                        try {
                            std::cout << "    Device #" << (j + 1) << ":\n";
                            platform::print_device_info(all_devices[j]);
                            std::cout << "\n";
                        } catch (const std::exception& e) {
                            std::cerr << "    Error getting info for device #" << (j + 1)
                                     << ": " << e.what() << "\n\n";
                        }
                    }
                }

                std::cout << "==========================================\n\n";

            } catch (const std::exception& e) {
                std::cerr << "Error processing platform #" << (i + 1) << ": " << e.what() << "\n";
                std::cerr << "Skipping to next platform...\n\n";
            }
        }

        std::cout << "OpenCL platform enumeration completed.\n";
        return 0;

    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument error: " << e.what() << "\n";
        return 1;
    } catch (const std::runtime_error& e) {
        std::cerr << "OpenCL runtime error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << "\n";
        return 1;
    }
}