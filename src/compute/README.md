# compute/ (Layer 2 — planned)

Typed launcher functions built on `Pipeline` (see architecture.md §5).
Each operation's `.cl` kernel source lives in `kernels/` next to its launcher.

Not implemented yet. When it is:

- `compute.hpp` / `compute.cpp` — declare/define split, same convention as
  `opencl_wrapper` (see `include/opencl_wrapper/platform.hpp` for the pattern).
- `kernels/vec_add.cl`, `kernels/mat_mul.cl`, `kernels/reduction.cl`,
  `kernels/prefix_sum.cl` — read from disk at runtime via `Program::from_file`.
- Uncomment the `compute` target in the root `CMakeLists.txt` once
  `compute.cpp` exists.
