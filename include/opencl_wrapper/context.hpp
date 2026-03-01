#pragma once

#include <CL/cl.h>
#include <CL/opencl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace context {
// Error helper
inline void check(cl_int err, const char *msg) {
  if (err != CL_SUCCESS)
    throw std::runtime_error(std::string(msg) + " (code " +
                             std::to_string(err) + ")");
}
// RAII context
struct context {
  cl_context handle = nullptr;
  context() = default;
  explicit context(const std::vector<cl_device_id> &devices,
                   cl_platform_id platform = nullptr,
                   void(CL_CALLBACK *pfn_notify)(const char *, const void *,
                                                 size_t, void *) = nullptr,
                   void *user_data = nullptr) {
    cl_int err = CL_SUCCESS;
    std::vector<cl_context_properties> props;
    if (platform) {
      props = {CL_CONTEXT_PLATFORM,
               reinterpret_cast<cl_context_properties>(platform), 0};
    }
    handle = clCreateContext(props.empty() ? nullptr : props.data(),
                             static_cast<cl_uint>(devices.size()),
                             devices.data(), pfn_notify, user_data, &err);
    check(err, "clCreateContext");
  }
  explicit context(cl_device_id device, cl_platform_id platform = nullptr)
      : context(std::vector<cl_device_id>{device}, platform) {}
  ~context() {
    if (handle)
      clReleaseContext(handle);
  }
  // Non copyable
  context(const context &) = delete;
  context &operator=(const context &) = delete;
  // Movable
  context(context &&o) noexcept : handle(o.handle) { o.handle = nullptr; }
  context &operator=(context &&o) noexcept {
    if (this != &o) {
      if (handle)
        clReleaseContext(handle);
      handle = o.handle;
      o.handle = nullptr;
    }
    return *this;
  }
  operator cl_context() const { return handle; }
  bool valid() const { return handle != nullptr; }
};
// RAII command queue
struct QUEUE {
  cl_command_queue handle = nullptr;
  QUEUE() = default;
  // OPENCL 3.0 style ---Uses clCreateCommandQueueWithProperties
  QUEUE(cl_context ctx, cl_device_id device,
        cl_command_queue_properties properties = 0) {
    cl_int err = CL_SUCCESS;
    const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, properties, 0};
    handle = clCreateCommandQueueWithProperties(
        ctx, device, properties ? props : nullptr, &err);
    check(err, "clCreateCommandQueueWithProperties");
  }
  ~QUEUE() {
    if (handle) {
      clFlush(handle);
      clFinish(handle);
      clReleaseCommandQueue(handle);
    }
  }
  QUEUE(const QUEUE &) = delete;
  QUEUE &operator=(const QUEUE &) = delete;
  QUEUE(QUEUE &&o) noexcept : handle(o.handle) { o.handle = nullptr; }
  QUEUE &operator=(QUEUE &&o) noexcept {
    if (this != &o) {
      if (handle) {
        clFlush(handle);
        clFinish(handle);
        clReleaseCommandQueue(handle);
      }
      handle = o.handle;
      o.handle = nullptr;
    }
    return *this;
  }
  operator cl_command_queue() const { return handle; }
  bool valid() const { return handle != nullptr; }
  void flush() const { clFlush(handle); }
  void finish() const { clFinish(handle); }
};
} // namespace context
