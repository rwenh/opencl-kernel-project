# opencl-kernel-project

OpenCL-based GPU acceleration and rendering toolkit written in modern C++17.

---

## Project Structure

```
opencl-kernel-project/
│
├── include/                            # Public headers (declarations only)
│   └── opencl_wrapper/
│       ├── platform.hpp                # Platform & device enumeration
│       ├── context.hpp                 # RAII context & command queue
│       ├── buffer.hpp                  # RAII device memory management
│       ├── program.hpp                 # Kernel compilation (source/file/binary/SPIR-V)
│       ├── dispatch.hpp                # NDRange, kernel dispatch, event profiling
│       └── pipeline.hpp                # High-level unified API
│
├── src/                                # Implementations
│   ├── opencl_wrapper/                 # Counterpart .cpp for each header
│   │   ├── platform.cpp
│   │   ├── context.cpp
│   │   ├── buffer.cpp
│   │   ├── program.cpp
│   │   ├── dispatch.cpp
│   │   └── pipeline.cpp
│   │
│   ├── compute/                        # GPU compute kernels
│   │   ├── kernels/                    # OpenCL kernel source files (.cl)
│   │   │   ├── vec_add.cl
│   │   │   ├── mat_mul.cl
│   │   │   ├── reduction.cl
│   │   │   └── prefix_sum.cl
│   │   └── compute.hpp                 # Compute kernel launcher API
│   │
│   ├── render/                         # Rendering pipeline (future)
│   │   ├── kernels/                    # Rendering kernel source files (.cl)
│   │   │   ├── raytracer.cl
│   │   │   └── rasterizer.cl
│   │   ├── image_buffer.hpp            # CL image2D/3D buffer wrapper
│   │   └── renderer.hpp                # Renderer API
│   │
│   └── examples/                       # Standalone example programs
│       ├── example.cpp                 # vec_add demo (pipeline smoke test)
│       ├── mat_mul_example.cpp         # Matrix multiply demo
│       └── render_example.cpp          # Rendering demo (future)
│
├── tests/                              # Unit & integration tests
│   ├── test_platform.cpp
│   ├── test_buffer.cpp
│   ├── test_dispatch.cpp
│   └── test_pipeline.cpp
│
├── docs/                               # Documentation
│   └── architecture.md                 # Design notes & API reference
│
├── CMakeLists.txt                      # Build system
├── .gitignore
├── LICENSE
└── README.md
```

---

## Modules

### `opencl_wrapper`
The core abstraction layer over the OpenCL C API. All resources are RAII-managed.

| Header | Responsibility |
|---|---|
| `platform.hpp` | Enumerate platforms/devices, query device info |
| `context.hpp` | Create OpenCL context and command queue |
| `buffer.hpp` | Allocate/read/write device buffers, async transfers |
| `program.hpp` | Compile kernels from source, file, binary, or SPIR-V IL |
| `dispatch.hpp` | Set kernel args, enqueue NDRange, event profiling |
| `pipeline.hpp` | Unified API — owns all resources in correct lifetime order |

### `compute`
Higher-level GPU compute operations built on top of the wrapper.
- Vector operations, matrix multiply, reductions, prefix sums
- Kernel `.cl` files live alongside their launchers

### `render` *(planned)*
GPU rendering pipeline.
- OpenCL image buffer support (`CL_MEM_OBJECT_IMAGE2D/3D`)
- Raytracer and rasterizer kernel implementations
- OpenGL/Vulkan interop

### `examples`
Standalone programs demonstrating usage of the toolkit.

### `tests`
Unit and integration tests for each wrapper module.

---

## Building

### Requirements
- CMake 3.16+
- C++17 compiler (GCC, Clang, MSVC)
- OpenCL SDK (e.g. Intel OpenCL, ROCm, NVIDIA CUDA toolkit)

### Steps
```bash
git clone https://github.com/rwenh/opencl-kernel-project.git
cd opencl-kernel-project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Status

| Module | Status |
|---|---|
| `opencl_wrapper` | In progress — bug fixes underway |
| `compute` | Planned |
| `render` | Planned |
| `examples` | In progress |
| `tests` | Planned |

---

## License

See [LICENSE](LICENSE).
