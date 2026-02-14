#include "opencl_wrapper/platform.hpp"
#include <iostream>

int main() {
    auto platforms = platform::get_platforms();
    std::cout << "Found " << platforms.size() << " platform(s)\n\n";

    for (auto& p : platforms) {
        platform::print_platform_info(p);

        auto devices = platform::get_devices(p);
        std::cout << "  Devices: " << devices.size() << "\n";

        for (auto& d : devices) {
            platform::print_device_info(d);
        }
        std::cout << "\n";
    }
}