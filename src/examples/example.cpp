// example.cpp
// Lists every opencl platform/device visibleon the
// system using the opencl_wrapper platform mod:
// kept deliberately small: only `platform` migrated
// to the declare/define split to be exercised;
// Implementation is going to happen for
// buffer/context/program/dispatch/pipeline and so on:

#include "opencl_wrapper/platform.hpp"

#include <iostream>

int main() {
	auto platforms = platform::get_platforms();
	if (platforms.empty()) {
		std::cout << "No Opencl platforms found on this system.\n";
		return 0;
	}
	for (auto p : platforms) {
		platform::print_platform_info(p);
		for(auto d: platform::get_devices(p))
			platform::print_device_info(d);
	}
	return 0;
}
