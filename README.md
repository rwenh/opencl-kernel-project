# opencl-kernel-project

OpenCL GPU acceleration and rendering toolkit, C++17. See `docs/architecture.md`
for the full design (layer model, module reference, resource lifetime rules).

## Layout

```
opencl-kernel-project/
├── CMakeLists.txt
├── include/
│   └── opencl_wrapper/       Layer 1 — public headers (declarations + templates)
│       ├── platform.hpp      ✅ migrated — reference pattern, see below
│       ├── buffer.hpp        header-only, pending migration
│       ├── context.hpp       header-only, pending migration
│       ├── program.hpp       header-only, pending migration
│       ├── dispatch.hpp      header-only, pending migration
│       └── pipeline.hpp      header-only, pending migration (facade over the above)
├── src/
│   ├── opencl_wrapper/       Layer 1 — .cpp definitions for migrated modules
│   │   └── platform.cpp      ✅ migrated
│   ├── compute/               Layer 2 — typed launchers (vec_add, mat_mul, ...) — planned
│   ├── render/                Layer 2 — image buffers / rendering — planned
│   └── examples/
│       └── example.cpp        lists all OpenCL platforms/devices
├── tests/
│   └── test_platform.cpp      raw-assert smoke test, no test framework
└── docs/
    └── architecture.md
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/example
```

Requires an OpenCL ICD loader + headers (`ocl-icd-opencl-dev` or vendor SDK)
and at least one ICD. For a GPU-less dev machine or CI, install PoCL for a
real CPU-backed platform:

```bash
sudo apt install pocl-opencl-icd
```

## The header/`.cpp` convention

Every `include/opencl_wrapper/*.hpp` file should hold:

- The class/struct **shape** — member variables and member function
  **signatures** (no bodies), *except* trivial one-liners (`bool valid() const
  { return handle != nullptr; }`, conversion operators) — those are fine
  staying inline in the class body.
- **Template** functions and methods, fully defined — they're instantiated
  per call site, so the body has to be visible wherever they're used.
- Free-function **declarations** for everything non-template.

The matching `src/opencl_wrapper/*.cpp` file holds the **out-of-line
definitions** for everything non-template. `platform.hpp` / `platform.cpp` is
the worked example — copy that split when migrating the next module.

`.cpp` files include their own header with the `opencl_wrapper/` prefix
(`#include "opencl_wrapper/buffer.hpp"`), matching
`target_include_directories(opencl_wrapper PUBLIC include)` in the root
`CMakeLists.txt`. Headers including their *siblings* (e.g. `pipeline.hpp`
pulling in `buffer.hpp`) use a bare quoted include, since same-directory
lookup resolves it directly.

A module only needs a `.cpp` once it actually has one — until then it stays
header-only (all `inline`) and isn't listed as an `opencl_wrapper` library
source in `CMakeLists.txt`. Header-only is a legitimate end state too, not
just a waypoint — see "Migration status" below.

## Migration status

| Module | Header | `.cpp` | Notes |
|---|---|---|---|
| `platform` | declare/define split | ✅ | reference pattern |
| `buffer` | header-only | — | still has the mislabeled `read()`/`write()` overload found in review — fix while migrating |
| `context` | header-only | — | clean; smallest lift |
| `program` | header-only | — | `build_program()`'s device/log branches are inverted — fix while migrating |
| `dispatch` | header-only | — | `enqueue()`'s `blocking` param is dead — drop it or wire it up |
| `pipeline` | header-only | — | picks `device` from *any* platform but pairs it with `platforms[0]` — fix device/platform selection together while migrating |

Pick a module, apply the `platform` pattern, uncomment its line in
`CMakeLists.txt`'s `opencl_wrapper` target, add its `tests/test_*.cpp`, done.
