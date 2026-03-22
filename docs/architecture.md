# Architecture & API Reference

**opencl-kernel-project** — OpenCL GPU acceleration and rendering toolkit, C++17.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Layer Model](#2-layer-model)
3. [Module Reference](#3-module-reference)
   - 3.1 [platform](#31-platform)
   - 3.2 [context](#32-context)
   - 3.3 [buffer](#33-buffer)
   - 3.4 [program](#34-program)
   - 3.5 [dispatch](#35-dispatch)
   - 3.6 [Pipeline](#36-pipeline)
4. [Resource Lifetime & Ownership](#4-resource-lifetime--ownership)
5. [Compute Layer](#5-compute-layer)
6. [Render Layer (Planned)](#6-render-layer-planned)
7. [Error Handling Strategy](#7-error-handling-strategy)
8. [Build System Layout](#8-build-system-layout)
9. [Testing Strategy](#9-testing-strategy)
10. [Design Decisions & Trade-offs](#10-design-decisions--trade-offs)
11. [Known Limitations & Future Work](#11-known-limitations--future-work)

---

## 1. Design Philosophy

The project is structured as three concentric layers, each building on the previous:

```
┌─────────────────────────────────────────┐
│          compute / render               │  High-level domain operations
│  (vec_add, mat_mul, raytracer, ...)     │
├─────────────────────────────────────────┤
│              Pipeline                   │  Unified facade — owns all resources
├─────────────────────────────────────────┤
│           opencl_wrapper                │  Thin, RAII-safe OpenCL C API wrappers
│  platform · context · buffer            │
│  program  · dispatch                    │
└─────────────────────────────────────────┘
              OpenCL C API
```

**Guiding rules:**

- **RAII everywhere.** Every `cl_*` handle is owned by a C++ struct with a well-defined destructor. Raw handles never escape into user code except through explicit conversion operators.
- **No silent failures.** Every OpenCL return code is checked. Errors throw `std::runtime_error` with the OpenCL error code and a human-readable context string.
- **Zero overhead at the wrapper layer.** `opencl_wrapper` structs are move-only value types. There is no heap allocation, virtual dispatch, or reference counting beyond what OpenCL itself manages internally.
- **Progressive abstraction.** You can use `dispatch::run()` directly against raw `cl_kernel` / `cl_command_queue` handles, or go all the way up to `Pipeline` and never touch an OpenCL type directly.
- **Headers declare, .cpp files define.** Template functions that need instantiation at the call site stay in headers. Everything else moves to `.cpp` to minimise recompilation and keep headers readable.

---

## 2. Layer Model

### Layer 0 — OpenCL C API (external)

The raw `cl_*` functions from `<CL/opencl.h>`. Never called directly by application code — always mediated through `opencl_wrapper`.

### Layer 1 — `opencl_wrapper` (include/opencl_wrapper/)

Thin, one-to-one RAII wrappers. Each module corresponds to a single OpenCL concept. They have no dependency on each other except `pipeline.hpp`, which aggregates all of them.

| Module | OpenCL concept wrapped |
|---|---|
| `platform` | `cl_platform_id`, `cl_device_id` |
| `context` | `cl_context`, `cl_command_queue` |
| `buffer` | `cl_mem` (buffer objects) |
| `program` | `cl_program`, `cl_kernel` |
| `dispatch` | `clEnqueueNDRangeKernel`, `cl_event` |
| `pipeline` | All of the above, composed |

### Layer 2 — `compute` / `render` (src/)

Domain-specific operations implemented using the wrapper. These modules own their `.cl` kernel files and expose typed C++ launcher functions. Users of the library primarily interact at this layer.

### Layer 3 — `examples` / application code

End-user code. Uses `Pipeline` or individual wrapper modules. Never touches OpenCL types directly.

---

## 3. Module Reference

### 3.1 `platform`

**Header:** `include/opencl_wrapper/platform.hpp`
**Source:** `src/opencl_wrapper/platform.cpp`
**Namespace:** `platform::`

Handles OpenCL platform and device enumeration. All functions are free functions — there is no platform RAII struct because `cl_platform_id` and `cl_device_id` are not owned handles (the runtime manages their lifetime).

#### Free functions

```cpp
// Enumeration
std::vector<cl_platform_id> get_platforms();
std::vector<cl_device_id>   get_devices(cl_platform_id, cl_device_type = CL_DEVICE_TYPE_ALL);
std::vector<cl_device_id>   get_all_devices(cl_device_type = CL_DEVICE_TYPE_ALL);
cl_device_id                select_best_device(cl_device_type preferred = CL_DEVICE_TYPE_GPU);

// String queries
std::string get_platform_name(cl_platform_id);
std::string get_device_name(cl_device_id);
std::string get_platform_info_str(cl_platform_id, cl_platform_info);
std::string get_device_info_str(cl_device_id, cl_device_info);

// Typed scalar queries
template<typename T>
T       get_device_info(cl_device_id, cl_device_info);
cl_uint get_compute_units(cl_device_id);
cl_ulong get_global_mem(cl_device_id);
cl_ulong get_local_mem(cl_device_id);
size_t  get_max_work_group_size(cl_device_id);
bool    supports_fp64(cl_device_id);

// Pretty printing
void print_platform_info(cl_platform_id, std::ostream& = std::cout);
void print_device_info(cl_device_id,    std::ostream& = std::cout);
```

#### Device selection heuristic

`select_best_device()` iterates all devices of the preferred type and returns the one with the highest `CL_DEVICE_MAX_COMPUTE_UNITS` count. If no devices of the preferred type exist, it falls back to `CL_DEVICE_TYPE_ALL`. This is a reasonable heuristic for discrete GPUs; it may select an iGPU over a dGPU on certain multi-GPU systems. For production use, consider exposing device selection to the caller.

---

### 3.2 `context`

**Header:** `include/opencl_wrapper/context.hpp`
**Source:** `src/opencl_wrapper/context.cpp`
**Namespace:** `context::`

RAII wrappers for `cl_context` and `cl_command_queue`.

#### `context::context`

```cpp
struct context {
    cl_context handle = nullptr;

    // Multi-device context with optional platform properties and error callback
    context(const std::vector<cl_device_id>& devices,
            cl_platform_id platform = nullptr,
            void(CL_CALLBACK* pfn_notify)(...) = nullptr,
            void* user_data = nullptr);

    // Single-device shortcut
    context(cl_device_id device, cl_platform_id platform = nullptr);

    // Non-copyable, movable
    operator cl_context() const;
    bool valid() const;
};
```

The `pfn_notify` callback, if provided, is called by the OpenCL runtime on context errors (useful for debugging driver issues). Pass `nullptr` in production.

#### `context::QUEUE`

```cpp
struct QUEUE {                          // TODO: rename to Queue in v2
    cl_command_queue handle = nullptr;

    // Uses clCreateCommandQueueWithProperties (OpenCL 2.0+ / 3.0)
    QUEUE(cl_context, cl_device_id, cl_command_queue_properties = 0);

    void flush()  const;   // clFlush  — non-blocking submission
    void finish() const;   // clFinish — blocks until queue drained

    operator cl_command_queue() const;
    bool valid() const;
};
```

Pass `CL_QUEUE_PROFILING_ENABLE` as `properties` to enable `dispatch::Event` timing. Pass `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` to allow the driver to reorder independent operations (requires explicit event dependencies).

> **Naming note:** `QUEUE` will be renamed to `Queue` in a future version to match C++ naming conventions and be consistent with the lowercase `context` struct. This is a breaking change deferred until the API stabilises.

---

### 3.3 `buffer`

**Header:** `include/opencl_wrapper/buffer.hpp`
**Source:** `src/opencl_wrapper/buffer.cpp`
**Namespace:** `buffer::`

RAII wrapper for `cl_mem` buffer objects. Supports typed and untyped transfers, async operations, sub-buffers, map/unmap, and device-side copies.

#### `buffer::Buffer`

```cpp
struct Buffer {
    cl_mem  handle    = nullptr;
    size_t  byte_size = 0;

    // Construction
    Buffer();                                                          // empty
    Buffer(cl_context, size_t bytes, cl_mem_flags = CL_MEM_READ_WRITE);
    template<typename T>
    Buffer(cl_context, const std::vector<T>& data, cl_mem_flags = CL_MEM_READ_WRITE);

    // Sub-buffer view (shares underlying memory)
    Buffer sub_buffer(size_t origin, size_t size,
                      cl_mem_flags = CL_MEM_READ_WRITE) const;

    // Blocking transfers
    void write(cl_command_queue, const void* data, size_t size=0, size_t offset=0) const;
    void read (cl_command_queue, void*       data, size_t size=0, size_t offset=0) const;

    // Typed blocking transfers (vector overloads)
    template<typename T>
    void write(cl_command_queue, const std::vector<T>& data, size_t offset=0) const;
    template<typename T>
    void read (cl_command_queue, std::vector<T>& data, size_t offset=0) const;
    template<typename T>
    std::vector<T> read(cl_command_queue) const;

    // Async transfers — caller owns the returned cl_event (wrap in dispatch::Event)
    cl_event write_async(cl_command_queue, const void*, size_t=0, size_t=0,
                         const std::vector<cl_event>& wait_list={}) const;
    cl_event read_async (cl_command_queue, void*,       size_t=0, size_t=0,
                         const std::vector<cl_event>& wait_list={}) const;

    // Device-side copy (no host involvement)
    void copy_to(cl_command_queue, const Buffer& dst,
                 size_t size=0, size_t src_offset=0, size_t dst_offset=0) const;

    // Pinned / mapped access
    void* map  (cl_command_queue, cl_map_flags = CL_MAP_READ|CL_MAP_WRITE,
                size_t offset=0, size_t size=0) const;
    void  unmap(cl_command_queue, void* ptr) const;

    operator cl_mem() const;
    bool   valid()    const;
    size_t size()     const;
};

// Free functions
void fill(cl_command_queue, const Buffer&, const void* pattern,
          size_t pattern_size, size_t offset=0, size_t size=0);
template<typename T>
void fill(cl_command_queue, const Buffer&, const T& value);
```

#### Transfer mode guidance

| Scenario | Recommended API |
|---|---|
| Small data, correctness critical | `write()` / `read()` (blocking) |
| Overlap transfer with compute | `write_async()` + `dispatch::Event` dependency |
| Zero-copy on unified memory (APU) | `map()` / `unmap()` |
| GPU → GPU copy (same context) | `copy_to()` |

#### Async event ownership

`write_async()` and `read_async()` return a raw `cl_event`. Always wrap the result in `dispatch::Event` to ensure `clReleaseEvent` is called:

```cpp
dispatch::Event ev{ buf.write_async(queue, data.data(), data.size() * sizeof(float)) };
// ... enqueue kernel ...
ev.wait();
```

---

### 3.4 `program`

**Header:** `include/opencl_wrapper/program.hpp`
**Source:** `src/opencl_wrapper/program.cpp`
**Namespace:** `program::`

Handles OpenCL program compilation from multiple source formats, and RAII kernel objects.

#### `program::Program`

```cpp
struct Program {
    cl_program handle = nullptr;

    // Factory methods (all static)
    static Program from_source(cl_context, const std::string& source,
                               const std::vector<cl_device_id>& = {},
                               const std::string& options = "");

    static Program from_file  (cl_context, const std::string& path,
                               const std::vector<cl_device_id>& = {},
                               const std::string& options = "");

    static Program from_il    (cl_context, const std::vector<uint8_t>& spirv,
                               const std::vector<cl_device_id>& = {},
                               const std::string& options = "");   // OpenCL 2.1+

    static Program from_binary(cl_context,
                               const std::vector<cl_device_id>&,
                               const std::vector<std::vector<uint8_t>>& binaries,
                               const std::string& options = "");

    // Binary extraction for caching
    std::vector<uint8_t> get_binary(cl_device_id) const;

    operator cl_program() const;
    bool valid() const;
};
```

#### `program::Kernel`

```cpp
struct Kernel {
    cl_kernel   handle = nullptr;
    std::string name;

    Kernel(cl_program, const std::string& kernel_name);

    // Argument setters
    template<typename T>
    void set_arg(cl_uint index, const T& value);
    void set_arg(cl_uint index, cl_mem buffer);          // specialisation
    void set_local_arg(cl_uint index, size_t bytes);     // local memory placeholder

    operator cl_kernel() const;
    bool valid() const;
};
```

#### Helper: `create_all_kernels`

```cpp
std::unordered_map<std::string, Kernel> create_all_kernels(cl_program);
```

Uses `clCreateKernelsInProgram` to introspect and wrap every kernel in a compiled program. Used by `Pipeline` when no explicit kernel name list is provided.

#### Compilation build options

Useful options to pass via the `options` string:

| Option | Effect |
|---|---|
| `-cl-std=CL3.0` | Target OpenCL 3.0 |
| `-cl-fast-relaxed-math` | Aggressive FP optimisations (breaks strict IEEE) |
| `-DBLOCK_SIZE=64` | Preprocessor define — pass kernel tuning constants |
| `-cl-mad-enable` | Allow fused multiply-add |
| `-g` | Generate debug info (vendor-dependent) |

#### Binary caching pattern

```cpp
// First run: compile and save
Program prog = Program::from_source(ctx, src, {device}, opts);
auto binary  = prog.get_binary(device);
// ... write binary to disk ...

// Subsequent runs: load from cache
// ... read binary from disk into std::vector<uint8_t> ...
Program prog = Program::from_binary(ctx, {device}, {binary}, opts);
```

---

### 3.5 `dispatch`

**Header:** `include/opencl_wrapper/dispatch.hpp`
**Source:** `src/opencl_wrapper/dispatch.cpp`
**Namespace:** `dispatch::`

NDRange construction, kernel enqueue, variadic argument setting, and event profiling.

#### `dispatch::NDRange`

```cpp
struct NDRange {
    size_t dims;
    size_t global[3];
    size_t local[3];   // 0 = let runtime choose

    explicit NDRange(size_t g, size_t l = 0);                        // 1D
    NDRange(size_t gx, size_t gy, size_t lx=0, size_t ly=0);        // 2D
    NDRange(size_t gx, size_t gy, size_t gz,
            size_t lx, size_t ly, size_t lz);                       // 3D

    const size_t* global_ptr() const;
    const size_t* local_ptr()  const;   // returns nullptr if all local == 0
};

// Round global up to the nearest multiple of local
size_t round_up(size_t global, size_t local);
```

#### `dispatch::enqueue`

```cpp
void enqueue(cl_command_queue,
             cl_kernel,
             const NDRange&,
             const std::vector<cl_event>& wait_list = {},
             cl_event* out_event = nullptr);
```

Bare enqueue with optional event dependency list. `clEnqueueNDRangeKernel` is always asynchronous — there is no blocking flag. Use `ev.wait()` or `queue.finish()` for synchronisation.

#### `dispatch::run` — variadic convenience

```cpp
template<typename... Args>
cl_event run(cl_command_queue, cl_kernel, const NDRange&, Args&&... args);
```

Sets all kernel arguments by index (0, 1, 2, ...) using `clSetKernelArg`, then calls `enqueue()`. Returns the raw `cl_event` — wrap in `dispatch::Event` for RAII lifetime and profiling.

```cpp
// Example
dispatch::Event ev{
    dispatch::run(queue, kernel, dispatch::NDRange{1024, 64},
                  buf_a.handle, buf_b.handle, buf_c.handle, N)
};
ev.wait();
std::cout << ev.elapsed_ms() << " ms\n";
```

**Argument passing rules:**
- `cl_mem` handles are passed as-is (use `buffer.handle` or the implicit `operator cl_mem()`)
- Scalar types (`int`, `float`, etc.) are passed by value — `sizeof(T)` bytes are copied
- Structs can be passed as kernel arguments if they are trivially copyable and the layout matches the `.cl` struct

#### `dispatch::Event`

```cpp
struct Event {
    cl_event handle = nullptr;

    explicit Event(cl_event);

    void wait() const;

    // Profiling (requires CL_QUEUE_PROFILING_ENABLE on the queue)
    cl_ulong queued()    const;   // nanoseconds since epoch
    cl_ulong submitted() const;
    cl_ulong started()   const;
    cl_ulong ended()     const;
    double   elapsed_ms() const;  // (ended - started) * 1e-6

    operator cl_event() const;
    bool valid() const;
};

void wait_all(const std::vector<cl_event>& events);
```

---

### 3.6 `Pipeline`

**Header:** `include/opencl_wrapper/pipeline.hpp`
**Source:** `src/opencl_wrapper/pipeline.cpp`
**Struct:** `Pipeline` (global namespace — intentionally not namespaced for ergonomics)

The unified high-level API. Owns all OpenCL resources in correct destruction order (kernels → program → queue → context). Most application code should only need `Pipeline`.

```cpp
struct Pipeline {
    cl_platform_id   platform = nullptr;
    cl_device_id     device   = nullptr;
    context::context ctx;
    context::QUEUE   queue;
    program::Program prog;
    std::unordered_map<std::string, program::Kernel> kernels;

    // From .cl file on disk
    explicit Pipeline(const std::string& kernel_path,
                      const std::vector<std::string>& kernel_names = {},
                      const std::string& build_options = "",
                      cl_command_queue_properties queue_props = 0);

    // From inline source string
    static Pipeline from_source(const std::string& source,
                                const std::vector<std::string>& kernel_names = {},
                                const std::string& build_options = "",
                                cl_command_queue_properties queue_props = 0);

    // Kernel lookup — throws if name not found
    program::Kernel& kernel(const std::string& name);

    // Buffer allocation scoped to this pipeline's context
    buffer::Buffer make_buffer(size_t bytes, cl_mem_flags = CL_MEM_READ_WRITE) const;
    template<typename T>
    buffer::Buffer make_buffer(const std::vector<T>& data,
                               cl_mem_flags = CL_MEM_READ_WRITE) const;

    void finish() const;   // drain the command queue

    // Non-copyable, movable
    Pipeline(Pipeline&&) noexcept = default;
    Pipeline& operator=(Pipeline&&) noexcept = default;
};
```

#### Typical usage pattern

```cpp
// 1. Build
auto p = Pipeline::from_source(kernel_src, {"my_kernel"}, "-cl-std=CL3.0",
                                CL_QUEUE_PROFILING_ENABLE);

// 2. Allocate buffers
auto d_input  = p.make_buffer(host_data, CL_MEM_READ_ONLY);
auto d_output = p.make_buffer(N * sizeof(float), CL_MEM_WRITE_ONLY);

// 3. Dispatch
dispatch::NDRange range{ dispatch::round_up(N, 64), 64 };
dispatch::Event ev{
    dispatch::run(p.queue, p.kernel("my_kernel"), range,
                  d_input.handle, d_output.handle, N)
};
ev.wait();

// 4. Retrieve
std::vector<float> result;
d_output.read(p.queue, result);

// 5. Cleanup: automatic (RAII)
```

---

## 4. Resource Lifetime & Ownership

OpenCL resources must be destroyed in the reverse order of their creation. `Pipeline` enforces this automatically through member declaration order and RAII destructors.

```
Creation order          Destruction order (reverse)
──────────────          ──────────────────────────
cl_platform_id          (not owned — runtime lifetime)
cl_device_id            (not owned — runtime lifetime)
cl_context         →    cl_kernel(s)   (Kernel dtors)
cl_command_queue   →    cl_program     (Program dtor)
cl_program         →    cl_command_queue (QUEUE dtor: flush+finish+release)
cl_kernel(s)       →    cl_context     (context dtor)
```

`Pipeline` member order in the struct definition deliberately matches creation order, so C++ destroys them in reverse — no manual ordering needed.

**Gotcha — sub-buffers:** A `Buffer::sub_buffer()` shares the parent `cl_mem`. The parent buffer **must outlive** all sub-buffers. `sub_buffer()` does not extend the parent's lifetime (no refcount increment). Prefer keeping sub-buffers in the same scope as their parent.

**Gotcha — async events:** `write_async()` / `read_async()` return raw `cl_event` handles. Always wrap them in `dispatch::Event` immediately. A dropped raw event leaks an OpenCL handle.

---

## 5. Compute Layer

**Location:** `src/compute/`
**Header:** `src/compute/compute.hpp`

Higher-level typed operations built on `Pipeline`. Each operation provides a C++ launcher function that handles buffer allocation, kernel dispatch, and result retrieval. Kernel source files (`.cl`) live alongside the launcher in `src/compute/kernels/`.

### Planned kernels

| File | Operation | Status |
|---|---|---|
| `kernels/vec_add.cl` | Element-wise `c[i] = a[i] + b[i]` for `float` | Planned |
| `kernels/mat_mul.cl` | Dense matrix multiplication (tiled, shared local memory) | Planned |
| `kernels/reduction.cl` | Parallel sum / max / min reduction | Planned |
| `kernels/prefix_sum.cl` | Exclusive prefix sum (scan) — Blelloch algorithm | Planned |

### `compute.hpp` API shape (proposed)

```cpp
namespace compute {

// Element-wise vector addition
// Result written into pre-allocated out buffer
void vec_add(Pipeline&, const buffer::Buffer& a,
             const buffer::Buffer& b, buffer::Buffer& c, int n);

// Matrix multiply: C = A * B
// A is (M x K), B is (K x N), C is (M x N) — row-major
void mat_mul(Pipeline&, const buffer::Buffer& A,
             const buffer::Buffer& B, buffer::Buffer& C,
             int M, int K, int N);

// Parallel reduction — returns single scalar result
float reduce_sum(Pipeline&, const buffer::Buffer& input, int n);
float reduce_max(Pipeline&, const buffer::Buffer& input, int n);

// Exclusive prefix sum (scan) — result in output, same size as input
void prefix_sum(Pipeline&, const buffer::Buffer& input,
                buffer::Buffer& output, int n);

} // namespace compute
```

### Kernel design conventions

All `.cl` kernels follow these conventions:

- First argument: `__global const T* input` (read-only sources)
- Last scalar argument: `int n` (element count) — always bounds-checked with `if (i < n)`
- Output buffers declared `__global T*` (never `const`) even if write-only
- Local memory passed as `__local T* smem` with size set via `Kernel::set_local_arg()`
- Work-group size hardcoded as a compile-time define (`-DBLOCK_SIZE=64`) for tuneability

---

## 6. Render Layer (Planned)

**Location:** `src/render/`

The render layer adds OpenCL image object support and a GPU rendering pipeline on top of `opencl_wrapper`.

### Planned files

| File | Responsibility |
|---|---|
| `image_buffer.hpp` | RAII wrapper for `cl_mem` image2D/3D objects |
| `renderer.hpp` | High-level render pipeline API |
| `kernels/raytracer.cl` | Path-tracing ray caster |
| `kernels/rasterizer.cl` | Triangle rasteriser |

### `image_buffer.hpp` design

```cpp
namespace render {

struct ImageBuffer {
    cl_mem   handle    = nullptr;
    size_t   width     = 0;
    size_t   height    = 0;
    cl_uint  channels  = 4;   // RGBA

    // 2D image
    ImageBuffer(cl_context, size_t width, size_t height,
                cl_channel_order = CL_RGBA,
                cl_channel_type  = CL_UNORM_INT8,
                cl_mem_flags     = CL_MEM_READ_WRITE);

    // 3D image (volume / texture array)
    ImageBuffer(cl_context, size_t width, size_t height, size_t depth,
                cl_channel_order = CL_RGBA,
                cl_channel_type  = CL_UNORM_INT8,
                cl_mem_flags     = CL_MEM_READ_WRITE);

    void read_to_host(cl_command_queue, void* dst) const;
    void write_from_host(cl_command_queue, const void* src);

    operator cl_mem() const;
    bool valid() const;
};

} // namespace render
```

### OpenGL / Vulkan interop (future)

Image buffers can be created from OpenGL textures or Vulkan images using `CL_MEM_OBJECT_IMAGE2D` with appropriate interop extensions (`cl_khr_gl_sharing`, `cl_khr_vulkan_event`). This requires the context to be created with interop properties — a separate `render::InteropContext` factory is planned.

---

## 7. Error Handling Strategy

Every module defines its own local `check()` helper:

```cpp
inline void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS)
        throw std::runtime_error(std::string(msg) + " (code " +
                                 std::to_string(err) + ")");
}
```

**Convention:** The error message always names the OpenCL function that failed, followed by the numeric error code in parentheses. For argument-indexed errors (e.g. `clSetKernelArg`) the index is included: `"clSetKernelArg[2] (code -48)"`.

**Common error codes:**

| Code | Constant | Typical cause |
|---|---|---|
| -4 | `CL_MEM_OBJECT_ALLOCATION_FAILURE` | Device out of memory |
| -5 | `CL_OUT_OF_RESOURCES` | Driver resource exhaustion |
| -11 | `CL_BUILD_PROGRAM_FAILURE` | Kernel syntax error — check build log |
| -30 | `CL_INVALID_VALUE` | Bad argument (size 0, null where not allowed) |
| -34 | `CL_INVALID_CONTEXT` | Context mismatch between resources |
| -38 | `CL_INVALID_MEM_OBJECT` | Invalid or released buffer handle |
| -48 | `CL_INVALID_KERNEL_ARGS` | Not all kernel args set before enqueue |
| -54 | `CL_INVALID_WORK_GROUP_SIZE` | Local size does not divide global size |

**Build failures** (`-11`) include the full compiler log in the exception message, printed per-device. Check stderr output when a `from_source()` / `from_file()` call throws.

**No error codes are swallowed.** Functions like `clFlush`, `clFinish`, and `clGetDeviceInfo` return `cl_int` but their results are not checked in the current implementation — this is a known gap (see §11).

---

## 8. Build System Layout

**File:** `CMakeLists.txt` (root)

The project builds as:

1. **`opencl_wrapper` static library** — compiled from `src/opencl_wrapper/*.cpp`, headers exposed from `include/`.
2. **`compute` static library** (planned) — compiled from `src/compute/compute.cpp`, links `opencl_wrapper`.
3. **`render` static library** (planned) — compiled from `src/render/*.cpp`, links `opencl_wrapper`.
4. **Example executables** — one per file in `src/examples/`, link `opencl_wrapper` (and `compute` / `render` as needed).
5. **Test executables** — one per file in `tests/`, link `opencl_wrapper`.

### Proposed CMakeLists.txt structure

```cmake
cmake_minimum_required(VERSION 3.16)
project(opencl_kernel_project CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

# ── Core wrapper library ──────────────────────────────────────────────────
add_library(opencl_wrapper STATIC
    src/opencl_wrapper/platform.cpp
    src/opencl_wrapper/context.cpp
    src/opencl_wrapper/buffer.cpp
    src/opencl_wrapper/program.cpp
    src/opencl_wrapper/dispatch.cpp
    src/opencl_wrapper/pipeline.cpp
)
target_include_directories(opencl_wrapper PUBLIC include/)
target_link_libraries(opencl_wrapper PUBLIC OpenCL::OpenCL)

# ── Compute library (planned) ─────────────────────────────────────────────
# add_library(compute STATIC src/compute/compute.cpp)
# target_link_libraries(compute PUBLIC opencl_wrapper)

# ── Examples ─────────────────────────────────────────────────────────────
add_executable(example src/examples/example.cpp)
target_link_libraries(example PRIVATE opencl_wrapper)

# ── Tests ────────────────────────────────────────────────────────────────
enable_testing()
foreach(test_src test_platform test_buffer test_dispatch test_pipeline)
    add_executable(${test_src} tests/${test_src}.cpp)
    target_link_libraries(${test_src} PRIVATE opencl_wrapper)
    add_test(NAME ${test_src} COMMAND ${test_src})
endforeach()
```

### .cl kernel files at runtime

OpenCL `.cl` source files are read from disk at runtime by `Program::from_file()`. They are **not** embedded in the binary. The executable must be run from the repository root, or the kernel path must be adjusted. Future improvement: embed kernel source as string literals using CMake's `configure_file()` or a `xxd`-generated header.

---

## 9. Testing Strategy

**Location:** `tests/`

Each wrapper module has a dedicated test file. Tests use raw assertions (`assert()` or a lightweight test macro) — no external test framework is required.

### Planned test files

| File | Coverage |
|---|---|
| `test_platform.cpp` | `get_platforms()`, `get_devices()`, `select_best_device()`, `print_device_info()` |
| `test_buffer.cpp` | Allocation, write/read round-trip, async transfers, sub-buffer, fill, copy_to |
| `test_dispatch.cpp` | NDRange construction, `round_up()`, `run()` variadic args, `Event` profiling |
| `test_pipeline.cpp` | `from_source()`, `from_file()`, `kernel()` lookup, `make_buffer()`, full vec_add round-trip |

### Testing without a GPU

Set `CL_DEVICE_TYPE_CPU` as the preferred type in `select_best_device()` for CI environments without a GPU. Most OpenCL SDKs ship a CPU runtime (Intel OpenCL CPU, Portable CL / `pocl`).

For headless CI, `pocl` (Portable OpenCL) is recommended:
```bash
sudo apt install pocl-opencl-icd
```

---

## 10. Design Decisions & Trade-offs

**Why not `std::shared_ptr` for OpenCL handles?**
OpenCL handles have their own internal reference counting (`clRetain*` / `clRelease*`). Adding a `shared_ptr` layer on top creates double-counting complexity and heap overhead. Move-only value semantics with explicit `clRelease*` in destructors is simpler and has zero overhead.

**Why is `Pipeline` in the global namespace?**
It is the primary entry point for most user code. Requiring `opencl_wrapper::Pipeline` everywhere is verbose. The design mirrors `std::vector` — a well-known type at the top level.

**Why does `dispatch::run()` return a raw `cl_event`?**
Returning `dispatch::Event` directly would require `<dispatch.hpp>` to know about itself recursively, and would force all callers to store an `Event` even when they don't need profiling. Returning the raw handle is explicit and lets callers choose: wrap it in `Event`, pass it to `wait_list`, or drop it if the operation is fire-and-forget (though dropping leaks the handle — documentation warning is the mitigation).

**Why no `Pipeline::add_kernel()`?**
Kernels are compiled as part of the program object; you cannot add a kernel from a different source to an existing program. The current design reflects this accurately — if you need more kernels, rebuild the pipeline with a different source.

---

## 11. Known Limitations & Future Work

| Area | Issue | Priority |
|---|---|---|
| `context::QUEUE` naming | All-caps `QUEUE` inconsistent with `context` | Low — rename to `Queue` in v2 |
| `clFlush` / `clFinish` return codes | Not checked — errors are silently dropped | Medium |
| Device selection | `select_best_device()` uses CU count only — may pick wrong device on multi-GPU systems | Medium |
| Event ownership | `run()` returns raw `cl_event` — easy to leak if not wrapped immediately | Medium |
| `.cl` file paths | Hardcoded relative paths — fragile when run outside repo root | Medium |
| Kernel binary caching | `get_binary()` exists but no cache-load-on-startup pattern implemented | Low |
| Multi-device dispatch | `Pipeline` supports only single device — no multi-GPU support | Low |
| OpenGL / Vulkan interop | No interop context factory yet | Low (render layer) |
| `cl_image` support | No `ImageBuffer` implementation yet | Blocked on render layer |
| Error context | `check()` has no stack context — hard to trace nested errors | Low |

---

*Last updated: session with opencl-kernel-project active development.*
