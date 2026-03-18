#pragma once

#include <CL/opencl.h>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace dispatch {
//-------Error helper----------------------------------------
inline void check(cl_int err, const char *msg) {
  if (err != CL_SUCCESS)
    throw std::runtime_error(std::string(msg) + "(code " + std::to_string(err) +
                             ")");
}
//------NDRange helper---------------------------------
struct NDRange {
  static constexpr size_t MAX_DIMS = 3;
  size_t dims = 0;
  size_t global[MAX_DIMS] = {1, 1, 1};
  size_t local[MAX_DIMS] = {0, 0, 0}; // 0= let runtime decide
  // 1D
  explicit NDRange(size_t g, size_t l = 0) : dims(1) {
    global[0] = g;
    local[0] = l;
  }
  // 2D
  NDRange(size_t gx, size_t gy, size_t lx = 0, size_t ly = 0) : dims(2) {
    global[0] = gx;
    global[1] = gy;
    local[0] = lx;
    local[1] = ly;
  }
  // 3D
  NDRange(size_t gx, size_t gy, size_t gz, size_t lx, size_t ly, size_t lz)
      : dims(3) {
    global[0] = gx;
    global[1] = gy;
    global[2] = gz;
    local[0] = lx;
    local[1] = ly;
    local[2] = lz;
  }
  const size_t *global_ptr() const { return global; }
  const size_t *local_ptr() const {
    // If all local dims are 0, return nullptr to let the runtime choose
    for (size_t i = 0; i < dims; ++i)
      if (local[i] != 0)
        return local;
    return nullptr;
  }
};
// Convenience : round global up to nearest mulitple of local
inline size_t round_up(size_t global, size_t local) {
  return local == 0 ? global : ((global + local - 1) / local) * local;
}
//----------Enqueue--------------------------------------------------------------------------------------
// -----------Basic Enqueue(Blocking by
// default)-------------------------------------
inline void enqueue(cl_command_queue queue, cl_kernel kernel,
                    const NDRange &range,
                    const std::vector<cl_event> &wait_list = {},
                    cl_event *out_event = nullptr, bool blocking = false) {
  cl_event ev = nullptr;
  check(clEnqueueNDRangeKernel(queue, kernel, static_cast<cl_uint>(range.dims),
                               nullptr, // global offset
                               range.global_ptr(), range.local_ptr(),
                               static_cast<cl_uint>(wait_list.size()),
                               wait_list.empty() ? nullptr : wait_list.data(),
                               out_event ? out_event : &ev),
        "clEnqueueNDRangeKernel");
}
//---------------Variadic argument setter  ---- sets arg before
//enqueue----------
namespace detail {
inline void set_args_impl(cl_kernel, cl_uint) {}
template <typename T, typename... Rest>
void set_args_impl(cl_kernel kernel, cl_uint idx, const T &val,
                   const Rest &...rest) {
  check(clSetKernelArg(kernel, idx, sizeof(T), &val),
        ("clSetKernelArg[" + std::to_string(idx) + "]").c_str());
  set_args_impl(kernel, idx + 1, rest...);
}
} // Namespace detail
// Set all kernel args and enqueue in one call
// dispatch::run(queue, kernel, NDRange{1024, 64}, buf.handle,  scalar, ...);
template <typename... Args>
cl_event run(cl_command_queue queue, cl_kernel kernel, const NDRange &range,
             Args &&...args) {
  detail::set_args_impl(kernel, 0, std::forward<Args>(args)...);
  cl_event ev = nullptr;
  enqueue(queue, kernel, range, {}, &ev);
  return ev; // Caller owns event
}
//---------Event
//helpers--------------------------------------------------------------------------------------
struct Event {
  cl_event handle = nullptr;
  Event() = default;
  explicit Event(cl_event e) : handle(e) {}
  ~Event() {
    if (handle)
      clReleaseEvent(handle);
  }

  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  Event(Event &&o) noexcept : handle(o.handle) { o.handle = nullptr; }
  Event &operator=(Event &&o) noexcept {
    if (this != &o) {
      if (handle)
        clReleaseEvent(handle);
      handle = o.handle;
      o.handle = nullptr;
    }
    return *this;
  }
  void wait() const {
    if (handle)
      check(clWaitForEvents(1, &handle), "clWaitForEvents");
  }
  // Profiling requires CL_QUEUE_PROFILING_ENABLE on the queue
  cl_ulong queued() const { return get_time(CL_PROFILING_COMMAND_QUEUED); }
  cl_ulong submitted() const { return get_time(CL_PROFILING_COMMAND_SUBMIT); }
  cl_ulong started() const { return get_time(CL_PROFILING_COMMAND_START); }
  cl_ulong ended() const { return get_time(CL_PROFILING_COMMAND_END); }
  double elapsed_ms() const {
    return static_cast<double>(ended() - started()) * 1e-6;
  }
  operator cl_event() const { return handle; }
  bool valid() const { return handle != nullptr; }

private:
  cl_ulong get_time(cl_profiling_info info) const {
    cl_ulong t = 0;
    check(clGetEventProfilingInfo(handle, info, sizeof(t), &t, nullptr),
          "clGetEventProfilingInfo");
    return t;
  }
};
inline void wait_all(const std::vector<cl_event> &events) {
  if (!events.empty())
    check(clWaitForEvents(static_cast<cl_uint>(events.size()), events.data()),
          "clWaitForEvents");
}
} // Namespace dispatch
