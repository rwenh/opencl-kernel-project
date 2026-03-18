#pragma once

//----------Unified Include------------------------------------------------
#include "buffer.hpp"
#include "context.hpp"
#include "dispatch.hpp"
#include "platform.hpp"
#include "program.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/*--------------------------Pipeline--------------------------------------------
 *  Owns all OpenCL resources in the correct lifetime order
 *  platform -> device -> context -> queue -> program -> kernels
 *
 *  USAGE:
 *  Pipeline p("kernel.cl", {"vecAdd", "matMul"});
 *  auto& add =p.kernel("vecAdd");
 *  auto ev = dispatch::run(p.queue, add, dispatch::NDRange{N, 64}, a,b,c ,N);
 */
struct Pipeline {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  context::context ctx;
  context::QUEUE queue;
  program::Program prog;
  std::unordered_map<std::string, program::Kernel> kernels;

  // Build from source file, select best GPU automatically
  explicit Pipeline(const std::string &kernel_path,
                    const std::vector<std::string> &kernel_names = {},
                    const std::string &build_options = "",
                    cl_command_queue_properties queue_props = 0) {
    // 1. Platform & device
    auto platforms = platform::get_platforms();
    if (platforms.empty())
      throw std::runtime_error("No OpenCL platforms found");
    platform = platforms[0];
    device = platform::select_best_device(CL_DEVICE_TYPE_GPU);

    // 2. Context & queue
    ctx = context::context(device, platform);
    queue = context::QUEUE(ctx, device, queue_props);
    // 3. Program
    prog =
        program::Program::from_file(ctx, kernel_path, {device}, build_options);
    // 4. Kernels
    if (kernel_names.empty()) {
      kernels = program::create_all_kernels(prog);
    } else {
      for (auto &name : kernel_names)
        kernels.emplace(name, program::Kernel(prog, name));
    }
  }
  // Build from inline source string
  static Pipeline from_source(const std::string &source,
                              const std::vector<std::string> &kernel_names = {},
                              const std::string &build_options = "",
                              cl_command_queue_properties queue_props = 0) {
    Pipeline p;
    auto platforms = platform::get_platforms();
    if (platforms.empty())
      throw std::runtime_error("No OpenCL platforms found");
    p.platform = platforms[0];
    p.device = platform::select_best_device(CL_DEVICE_TYPE_GPU);
    p.ctx = context::context(p.device, p.platform);
    p.queue = context::QUEUE(p.ctx, p.device, queue_props);
    p.prog =
        program::Program::from_source(p.ctx, source, {p.device}, build_options);
    if (kernel_names.empty()) {
      p.kernels = program::create_all_kernels(p.prog);
    } else {
      for (auto &name : kernel_names)
        p.kernels.emplace(name, program::Kernel(p.prog, name));
    }
    return p;
  }
  program::Kernel &kernel(const std::string &name) {
    auto it = kernels.find(name);
    if (it == kernels.end())
      throw std::runtime_error("Kernel not found : " + name);
    return it->second;
  }
  // Convenience : allocate and device buffer scoped to this pipeline's context
  buffer::Buffer make_buffer(size_t bytes,
                             cl_mem_flags flags = CL_MEM_READ_WRITE) const {
    return buffer::Buffer(ctx, bytes, flags);
  }
  template <typename T>
  buffer::Buffer make_buffer(const std::vector<T> &data,
                             cl_mem_flags flags = CL_MEM_READ_WRITE) const {
    return buffer::Buffer(ctx, data, flags);
  }
  void finish() const { queue.finish(); }

private:
  Pipeline() = default;
};
