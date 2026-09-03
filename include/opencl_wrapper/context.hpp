#pragma once

#include <CL/opencl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace context {

inline void check(cl_int err, const char *msg) {
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string(msg) + " (code " +
                                 std::to_string(err) + ")");
}

struct context {
    cl_context handle = nullptr;
    context() = default;

    explicit context(const std::vector<cl_device_id> &devices,
                     cl_platform_id platform = nullptr,
                     void (CL_CALLBACK *pfn_notify)(const char *, const void *,
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

    context(const context &) = delete;
    context &operator=(const context &) = delete;

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

struct Queue {
    cl_command_queue handle = nullptr;
    Queue() = default;

    Queue(cl_context ctx, cl_device_id device,
          cl_command_queue_properties properties = 0) {
        cl_int err = CL_SUCCESS;
        const cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, properties, 0};
        handle = clCreateCommandQueueWithProperties(
            ctx, device, properties ? props : nullptr, &err);
        check(err, "clCreateCommandQueueWithProperties");
    }

    ~Queue() {
        if (handle) {
            clFlush(handle);
            clFinish(handle);
            clReleaseCommandQueue(handle);
        }
    }

    Queue(const Queue &) = delete;
    Queue &operator=(const Queue &) = delete;

    Queue(Queue &&o) noexcept : handle(o.handle) { o.handle = nullptr; }
    Queue &operator=(Queue &&o) noexcept {
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

using QUEUE = Queue; // Compatibility typedef

} // namespace context
