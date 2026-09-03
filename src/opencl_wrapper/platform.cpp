#include "opencl_wrapper/platform.hpp"
#include <stdexcept>

namespace platform {

void check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string(msg) + " (code " +
                                 std::to_string(err) + ") ");
}

std::vector<cl_platform_id> get_platforms() {
    cl_uint count = 0;
    check(clGetPlatformIDs(0, nullptr, &count), "clGetPlatformIDs count");
    if (count == 0)
        return {};
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr), "clGetPlatformIDs");
    return platforms;
}

std::vector<cl_device_id> get_devices(cl_platform_id platform, cl_device_type type) {
    cl_uint count = 0;
    cl_int err = clGetDeviceIDs(platform, type, 0, nullptr, &count);
    if (err == CL_DEVICE_NOT_FOUND || count == 0)
        return {};
    check(err, "clGetDeviceIDs count");
    std::vector<cl_device_id> devices(count);
    check(clGetDeviceIDs(platform, type, count, devices.data(), nullptr),
          "clGetDeviceIDs");
    return devices;
}

std::vector<cl_device_id> get_all_devices(cl_device_type type) {
    std::vector<cl_device_id> all;
    for (auto p : get_platforms()) {
        auto devs = get_devices(p, type);
        all.insert(all.end(), devs.begin(), devs.end());
    }
    return all;
}

cl_device_id select_best_device(cl_device_type preferred) {
    auto devs = get_all_devices(preferred);
    if (devs.empty())
        devs = get_all_devices(CL_DEVICE_TYPE_ALL);
    if (devs.empty())
        throw std::runtime_error("No OpenCL devices found ");
    cl_device_id best = devs[0];
    cl_uint best_cu = 0;
    for (auto d : devs) {
        cl_uint cu = 0;
        clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
        if (cu > best_cu) {
            best_cu = cu;
            best = d;
        }
    }
    return best;
}

std::string get_platform_info_str(cl_platform_id platform, cl_platform_info param) {
    size_t size = 0;
    clGetPlatformInfo(platform, param, 0, nullptr, &size);
    if (size == 0) return "";
    std::string result(size, '\0');
    clGetPlatformInfo(platform, param, size, result.data(), nullptr);
    if (!result.empty() && result.back() == '\0')
        result.pop_back();
    return result;
}

std::string get_device_info_str(cl_device_id device, cl_device_info param) {
    size_t size = 0;
    clGetDeviceInfo(device, param, 0, nullptr, &size);
    if (size == 0) return "";
    std::string result(size, '\0');
    clGetDeviceInfo(device, param, size, result.data(), nullptr);
    if (!result.empty() && result.back() == '\0')
        result.pop_back();
    return result;
}

std::string get_platform_name(cl_platform_id platform) {
    return get_platform_info_str(platform, CL_PLATFORM_NAME);
}

std::string get_device_name(cl_device_id device) {
    return get_device_info_str(device, CL_DEVICE_NAME);
}

cl_uint get_compute_units(cl_device_id d) {
    return get_device_info<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS);
}

cl_ulong get_global_mem(cl_device_id d) {
    return get_device_info<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_SIZE);
}

cl_ulong get_local_mem(cl_device_id d) {
    return get_device_info<cl_ulong>(d, CL_DEVICE_LOCAL_MEM_SIZE);
}

size_t get_max_work_group_size(cl_device_id d) {
    return get_device_info<size_t>(d, CL_DEVICE_MAX_WORK_GROUP_SIZE);
}

bool supports_fp64(cl_device_id d) {
    auto ext = get_device_info_str(d, CL_DEVICE_EXTENSIONS);
    return ext.find("cl_khr_fp64") != std::string::npos;
}

void print_platform_info(cl_platform_id platform, std::ostream &out) {
    out << "Platform : " << get_platform_info_str(platform, CL_PLATFORM_NAME) << "\n"
        << "Vendor   : " << get_platform_info_str(platform, CL_PLATFORM_VENDOR) << "\n"
        << "Version  : " << get_platform_info_str(platform, CL_PLATFORM_VERSION) << "\n";
}

void print_device_info(cl_device_id device, std::ostream &out) {
    out << "Device   : " << get_device_name(device) << "\n"
        << "CUs      : " << get_compute_units(device) << "\n"
        << "GMem     : " << get_global_mem(device) / (1024ULL * 1024ULL) << " MB\n"
        << "LMem     : " << get_local_mem(device) / 1024ULL << " KB\n"
        << "WGSize   : " << get_max_work_group_size(device) << "\n"
        << "FP64     : " << (supports_fp64(device) ? "yes" : "no") << "\n";
}

} // namespace platform
