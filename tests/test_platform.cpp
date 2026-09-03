#include "opencl_wrapper/platform.hpp"
#include <cassert>
#include <iostream>

int main() {
    auto platforms = platform::get_platforms();
    std::cout << "Found " << platforms.size() << " OpenCL platform(s)\n";

    if (platforms.empty()) {
        std::cout << "No OpenCL platforms available in this environment -- "
                     "skipping device-level checks.\n";
        return 0;
    }
    for (auto p : platforms) {
        platform::print_platform_info(p);
        auto devices = platform::get_devices(p);
        std::cout << "  " << devices.size() << " device(s)\n";
        for (auto d : devices) {
            platform::print_device_info(d);
            assert(platform::get_compute_units(d) > 0);
            assert(platform::get_max_work_group_size(d) > 0);
        }
    }
    cl_device_id best = platform::select_best_device(CL_DEVICE_TYPE_ALL);
    assert(best != nullptr);
    std::cout << "Best device: " << platform::get_device_name(best) << "\n";

    std::cout << "test_platform: all checks passed\n";
    return 0;
}
