# render/ (Layer 2 — planned)

OpenCL image-object support and a GPU rendering pipeline on top of
`opencl_wrapper` (see architecture.md §6). Blocked on `cl_image` support.

Not implemented yet. When it is:

- `image_buffer.hpp` / `.cpp` — RAII wrapper for `cl_mem` image2D/3D objects.
- `renderer.hpp` / `.cpp` — high-level render pipeline API.
- `kernels/raytracer.cl`, `kernels/rasterizer.cl`.
- Uncomment the `render` target in the root `CMakeLists.txt` once
  `renderer.cpp` exists.
