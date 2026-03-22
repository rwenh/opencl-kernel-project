# opencl-kernel-project

OpenCL-based GPU acceleration and rendering toolkit written in modern C++17.

> **Status:** `opencl_wrapper` layer complete (bug fixes applied). `compute` and `render` layers planned. See [docs/architecture.md](docs/architecture.md) for the full design reference.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [File Inventory](#file-inventory)
3. [Module Overview](#module-overview)
4. [Dependency Graph](#dependency-graph)
5. [What Needs to Be Written](#what-needs-to-be-written)
6. [Building](#building)
7. [Running the Example](#running-the-example)
8. [Status](#status)
9. [License](#license)

---

## Project Structure

```
opencl-kernel-project/
│
├── include/                                    # Public headers — declarations only
│   └── opencl_wrapper/                         # Core OpenCL abstraction layer
│       ├── platform.hpp                        # Platform & device enumeration
│       ├── context.hpp                         # RAII context & command queue
│       ├── buffer.hpp                          # RAII device memory (cl_mem)
│       ├── program.hpp                         # Kernel compilation: source/file/binary/SPIR-V
│       ├── dispatch.hpp                        # NDRange, kernel dispatch, event profiling
│       └── pipeline.hpp                        # High-level facade — owns all resources
│
├── src/                                        # All implementation files
│   │
│   ├── opencl_wrapper/                         # .cpp counterparts for every header
│   │   ├── platform.cpp                        #  ↳ platform & device queries
│   │   ├── context.cpp                         #  ↳ context + queue constructors/dtors
│   │   ├── buffer.cpp                          #  ↳ buffer alloc, transfers, map/unmap
│   │   ├── program.cpp                         #  ↳ compilation, binary cache, kernel wrap
│   │   ├── dispatch.cpp                        #  ↳ enqueue, event helpers, profiling
│   │   └── pipeline.cpp                        #  ↳ Pipeline ctor, from_source factory
│   │
│   ├── compute/                                # [PLANNED] GPU compute operations
│   │   ├── kernels/                            # OpenCL kernel source files
│   │   │   ├── vec_add.cl                      #   Element-wise float vector addition
│   │   │   ├── mat_mul.cl                      #   Tiled matrix multiply (shared local mem)
│   │   │   ├── reduction.cl                    #   Parallel sum / max / min reduction
│   │   │   └── prefix_sum.cl                   #   Exclusive prefix sum (Blelloch scan)
│   │   ├── compute.hpp                         #   C++ launcher API for all kernels
│   │   └── compute.cpp                         #   Launcher implementations
│   │
│   ├── render/                                 # [PLANNED] GPU rendering pipeline
│   │   ├── kernels/                            # Rendering kernel source files
│   │   │   ├── raytracer.cl                    #   Path-tracing ray caster
│   │   │   └── rasterizer.cl                   #   Triangle rasteriser
│   │   ├── image_buffer.hpp                    #   RAII cl_image2D / cl_image3D wrapper
│   │   ├── image_buffer.cpp                    #   Image buffer implementation
│   │   ├── renderer.hpp                        #   High-level render pipeline API
│   │   └── renderer.cpp                        #   Renderer implementation
│   │
│   └── examples/                               # Standalone demo programs
│       ├── example.cpp                         #   vec_add smoke test (pipeline end-to-end)
│       ├── mat_mul_example.cpp                 #   [PLANNED] Matrix multiply demo
│       └── render_example.cpp                  #   [PLANNED] Rendering demo
│
├── tests/                                      # Unit & integration tests
│   ├── test_platform.cpp                       # [PLANNED] Platform/device enumeration tests
│   ├── test_buffer.cpp                         # [PLANNED] Buffer alloc, transfer, sub-buffer
│   ├── test_dispatch.cpp                       # [PLANNED] NDRange, run(), Event profiling
│   └── test_pipeline.cpp                       # [PLANNED] Full pipeline round-trip tests
│
├── docs/                                       # Documentation
│   └── architecture.md                         # Design notes, layer model, full API reference
│
├── CMakeLists.txt                              # Build system (see Building section)
├── .gitignore
├── LICENSE
└── README.md
```

---

## File Inventory

A complete count of every file in the project, by category and status.

### Headers (include/) — 6 files, all complete

| File | Namespace / Struct | Lines (approx) | Status |
|---|---|---|---|
| `include/opencl_wrapper/platform.hpp` | `platform::` | ~110 | ✅ Done |
| `include/opencl_wrapper/context.hpp` | `context::` | ~90 | ✅ Done |
| `include/opencl_wrapper/buffer.hpp` | `buffer::` | ~180 | ✅ Done |
| `include/opencl_wrapper/program.hpp` | `program::` | ~180 | ✅ Done |
| `include/opencl_wrapper/dispatch.hpp` | `dispatch::` | ~150 | ✅ Done |
| `include/opencl_wrapper/pipeline.hpp` | `Pipeline` | ~100 | ✅ Done |

### Wrapper implementations (src/opencl_wrapper/) — 6 files, all complete

| File | What it implements | Bugs fixed | Status |
|---|---|---|---|
| `src/opencl_wrapper/platform.cpp` | `get_platforms`, `get_devices`, `print_device_info` | Missing `\n` between CUs/GMem | ✅ Done |
| `src/opencl_wrapper/context.cpp` | `context` and `QUEUE` constructors/dtors | `properties==0` branch cleanup | ✅ Done |
| `src/opencl_wrapper/buffer.cpp` | All `Buffer` methods, `fill()` | `read()` calling `write()` internally; 2 typos in error strings | ✅ Done |
| `src/opencl_wrapper/program.cpp` | Factory methods, `Kernel`, `create_all_kernels` | Inverted build-log condition; 2 typos in error strings | ✅ Done |
| `src/opencl_wrapper/dispatch.cpp` | `enqueue`, `Event`, `wait_all` | `blocking` param unused; error string spacing | ✅ Done |
| `src/opencl_wrapper/pipeline.cpp` | `Pipeline` ctor, `from_source`, `kernel()`, `make_buffer` | Missing explicit move ops | ✅ Done |

### Compute layer (src/compute/) — 6 files needed, 0 written

| File | Type | Description | Status |
|---|---|---|---|
| `src/compute/compute.hpp` | C++ header | Typed launcher API for all kernels | 🔲 Planned |
| `src/compute/compute.cpp` | C++ source | Launcher implementations | 🔲 Planned |
| `src/compute/kernels/vec_add.cl` | OpenCL kernel | `c[i] = a[i] + b[i]`, float | 🔲 Planned |
| `src/compute/kernels/mat_mul.cl` | OpenCL kernel | Tiled matrix multiply, shared local memory | 🔲 Planned |
| `src/compute/kernels/reduction.cl` | OpenCL kernel | Parallel sum / max / min | 🔲 Planned |
| `src/compute/kernels/prefix_sum.cl` | OpenCL kernel | Exclusive scan (Blelloch algorithm) | 🔲 Planned |

### Render layer (src/render/) — 6 files needed, 0 written

| File | Type | Description | Status |
|---|---|---|---|
| `src/render/image_buffer.hpp` | C++ header | RAII `cl_image2D` / `cl_image3D` wrapper | 🔲 Planned |
| `src/render/image_buffer.cpp` | C++ source | Image buffer implementation | 🔲 Planned |
| `src/render/renderer.hpp` | C++ header | Render pipeline API | 🔲 Planned |
| `src/render/renderer.cpp` | C++ source | Renderer implementation | 🔲 Planned |
| `src/render/kernels/raytracer.cl` | OpenCL kernel | Path-tracing ray caster | 🔲 Planned |
| `src/render/kernels/rasterizer.cl` | OpenCL kernel | Triangle rasteriser | 🔲 Planned |

### Examples (src/examples/) — 3 files, 1 complete

| File | Depends on | Description | Status |
|---|---|---|---|
| `src/examples/example.cpp` | `opencl_wrapper` | vec_add smoke test | ✅ Done (bug fixed) |
| `src/examples/mat_mul_example.cpp` | `opencl_wrapper`, `compute` | Matrix multiply demo | 🔲 Planned |
| `src/examples/render_example.cpp` | `opencl_wrapper`, `render` | Rendering demo | 🔲 Planned |

### Tests (tests/) — 4 files needed, 0 written

| File | Tests | Status |
|---|---|---|
| `tests/test_platform.cpp` | `get_platforms`, `get_devices`, `select_best_device`, `print_device_info` | 🔲 Planned |
| `tests/test_buffer.cpp` | Alloc, write/read round-trip, async, sub-buffer, fill, copy_to | 🔲 Planned |
| `tests/test_dispatch.cpp` | NDRange, `round_up`, `run()`, `Event` profiling | 🔲 Planned |
| `tests/test_pipeline.cpp` | `from_source`, `from_file`, `kernel()` lookup, full vec_add round-trip | 🔲 Planned |

### Build & Docs — 3 files, all present

| File | Status |
|---|---|
| `CMakeLists.txt` | ⚠️ Needs update (see Building section) |
| `docs/architecture.md` | ✅ Done |
| `README.md` | ✅ This file |

---

### Total file count summary

| Category | Files needed | Files complete | Files remaining |
|---|---|---|---|
| Headers (`include/`) | 6 | 6 | 0 |
| Wrapper `.cpp` (`src/opencl_wrapper/`) | 6 | 6 | 0 |
| Compute layer (`src/compute/`) | 6 | 0 | **6** |
| Render layer (`src/render/`) | 6 | 0 | **6** |
| Examples (`src/examples/`) | 3 | 1 | **2** |
| Tests (`tests/`) | 4 | 0 | **4** |
| Build & docs | 3 | 3 | 0 |
| **Total** | **34** | **17** | **18** |

---

## Module Overview

### `opencl_wrapper` — Layer 1 (complete)

The core abstraction layer over the OpenCL C API. Every `cl_*` handle is owned by a C++ RAII struct. No raw handles escape into application code. All resources are move-only (non-copyable). Errors throw `std::runtime_error`.

| Header | Struct / Namespace | OpenCL concept |
|---|---|---|
| `platform.hpp` | `platform::` | `cl_platform_id`, `cl_device_id` |
| `context.hpp` | `context::context`, `context::QUEUE` | `cl_context`, `cl_command_queue` |
| `buffer.hpp` | `buffer::Buffer` | `cl_mem` (buffer objects) |
| `program.hpp` | `program::Program`, `program::Kernel` | `cl_program`, `cl_kernel` |
| `dispatch.hpp` | `dispatch::NDRange`, `dispatch::Event` | `clEnqueueNDRangeKernel`, `cl_event` |
| `pipeline.hpp` | `Pipeline` | All of the above, composed |

### `compute` — Layer 2 (planned)

Typed C++ launchers for common GPU compute operations. Each function accepts a `Pipeline&` and typed buffer arguments; kernel dispatch details are hidden. The `.cl` kernel files handle the actual GPU work.

Planned operations: `vec_add`, `mat_mul`, `reduce_sum`, `reduce_max`, `prefix_sum`.

### `render` — Layer 2 (planned)

GPU rendering pipeline using OpenCL image objects (`cl_image2D`, `cl_image3D`). Adds an `ImageBuffer` RAII wrapper and a `Renderer` API over the `opencl_wrapper` layer. OpenGL/Vulkan interop deferred to a later milestone.

### `examples`

Standalone programs that demonstrate real usage. Not part of the library — link against it. Each example exercises a specific layer of the stack.

### `tests`

Per-module tests covering allocation, transfer correctness, dispatch, and pipeline round-trips. Designed to run on CPU OpenCL runtimes (e.g. `pocl`) for headless CI.

---

## Dependency Graph

```
example.cpp / mat_mul_example.cpp / render_example.cpp
        │
        ▼
    Pipeline          ←── single entry point for most user code
        │
   ┌────┴────────────────────────────────┐
   ▼        ▼          ▼        ▼        ▼
platform  context    buffer  program  dispatch
   │        │          │        │        │
   └────────┴──────────┴────────┴────────┘
                       │
                  OpenCL C API
                 <CL/opencl.h>

compute.hpp  ──────────────────────────▶  Pipeline + buffer::Buffer
render/renderer.hpp  ──────────────────▶  Pipeline + render::ImageBuffer
render/image_buffer.hpp  ──────────────▶  opencl_wrapper (context, buffer internals)
```

**Include rules:**
- `pipeline.hpp` includes all other `opencl_wrapper` headers — include only `pipeline.hpp` in application code.
- `compute.hpp` includes `pipeline.hpp` and `buffer.hpp`.
- `render/renderer.hpp` includes `pipeline.hpp` and `render/image_buffer.hpp`.
- Test files include individual module headers directly to avoid over-linking.

---

## What Needs to Be Written

The following is the ordered work remaining, from shortest to longest dependency chain.

### Step 1 — Tests (no new dependencies)

Write the 4 test files in `tests/`. These only depend on `opencl_wrapper`, which is already complete. Running them will validate all bug fixes before compute/render work begins.

```
tests/test_platform.cpp    ~  80 lines
tests/test_buffer.cpp      ~ 150 lines
tests/test_dispatch.cpp    ~ 120 lines
tests/test_pipeline.cpp    ~ 200 lines
```

### Step 2 — Compute kernels (.cl files)

Write the 4 OpenCL kernel source files. These are pure C99 OpenCL C — no C++ dependencies.

```
src/compute/kernels/vec_add.cl      ~  20 lines
src/compute/kernels/reduction.cl    ~  60 lines  (two-pass: local then global)
src/compute/kernels/prefix_sum.cl   ~  80 lines  (Blelloch up-sweep / down-sweep)
src/compute/kernels/mat_mul.cl      ~ 50 lines  (tiled, uses __local)
```

### Step 3 — Compute launcher (depends on Step 2)

Write `compute.hpp` and `compute.cpp`. These wrap the `.cl` files with typed C++ APIs.

```
src/compute/compute.hpp    ~  50 lines
src/compute/compute.cpp    ~ 150 lines
```

### Step 4 — mat_mul example (depends on Step 3)

```
src/examples/mat_mul_example.cpp    ~ 100 lines
```

### Step 5 — Render image buffer (no dependency on compute)

```
src/render/image_buffer.hpp    ~  80 lines
src/render/image_buffer.cpp    ~ 120 lines
```

### Step 6 — Render kernels (.cl files, depends on Step 5 API)

```
src/render/kernels/rasterizer.cl    ~ 120 lines
src/render/kernels/raytracer.cl     ~ 200 lines
```

### Step 7 — Renderer (depends on Steps 5–6)

```
src/render/renderer.hpp    ~  60 lines
src/render/renderer.cpp    ~ 180 lines
```

### Step 8 — Render example (depends on Step 7)

```
src/examples/render_example.cpp    ~ 120 lines
```

### Step 9 — CMakeLists.txt update

Add targets for `compute`, `render`, new examples, and test runner. Approximately 40 additional lines.

---

## Building

### Requirements

| Dependency | Version | Notes |
|---|---|---|
| CMake | 3.16+ | Build system |
| C++ compiler | GCC 8+, Clang 7+, MSVC 2019+ | Must support C++17 |
| OpenCL SDK | Any | Intel OpenCL, ROCm, NVIDIA CUDA toolkit |
| OpenCL ICD loader | Any | Usually bundled with SDK |

For headless / CI environments without a GPU, install `pocl` (Portable OpenCL):
```bash
sudo apt install pocl-opencl-icd   # Ubuntu/Debian
brew install pocl                  # macOS
```

### Build steps

```bash
git clone https://github.com/rwenh/opencl-kernel-project.git
cd opencl-kernel-project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

For a debug build with symbols:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Build targets

| Target | Command | Output |
|---|---|---|
| Wrapper library | `make opencl_wrapper` | `libopencl_wrapper.a` |
| vec_add example | `make example` | `./example` |
| All tests | `make && ctest` | Pass/fail per test |

---

## Running the Example

```bash
cd build
./example
```

Expected output:
```
Device   : <your GPU name>
CUs      : 48
GMem     : 8192 MB
LMem     : 64 KB
WGSize   : 1024
FP64     : yes

Elapsed  : 0.042 ms
PASS
```

The example runs a 1024-element float vector addition (`c[i] = a[i] + b[i]`), times the kernel with OpenCL profiling events, reads back the result, and verifies correctness.

---

## Status

| Component | Files | Status | Notes |
|---|---|---|---|
| `opencl_wrapper` headers | 6 | ✅ Complete | Bug fixes applied |
| `opencl_wrapper` sources | 6 | ✅ Complete | All .cpp files written |
| `examples/example.cpp` | 1 | ✅ Complete | Kernel const-bug fixed, stray brace removed |
| `tests/` | 4 | 🔲 Planned | Step 1 — no new dependencies |
| `compute/` kernels (.cl) | 4 | 🔲 Planned | Step 2 |
| `compute/` C++ launcher | 2 | 🔲 Planned | Step 3 |
| `examples/mat_mul_example.cpp` | 1 | 🔲 Planned | Step 4 |
| `render/image_buffer` | 2 | 🔲 Planned | Step 5 |
| `render/` kernels (.cl) | 2 | 🔲 Planned | Step 6 |
| `render/renderer` | 2 | 🔲 Planned | Step 7 |
| `examples/render_example.cpp` | 1 | 🔲 Planned | Step 8 |
| `CMakeLists.txt` | 1 | ⚠️ Partial | Needs compute/render/test targets |
| `docs/architecture.md` | 1 | ✅ Complete | Full API reference |

---

## License

See [LICENSE](LICENSE).
