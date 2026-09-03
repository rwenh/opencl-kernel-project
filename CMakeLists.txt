cmake_minimum_required(VERSION 3.16)
project(opencl_kernel_project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(OpenCL REQUIRED)

# ───────────────────────── opencl_wrapper (Layer 1) ─────────────────────────
add_library(opencl_wrapper STATIC
    src/opencl_wrapper/platform.cpp
    # src/opencl_wrapper/context.cpp    # TODO: migrate
    # src/opencl_wrapper/buffer.cpp     # TODO: migrate
    # src/opencl_wrapper/program.cpp    # TODO: migrate
    # src/opencl_wrapper/dispatch.cpp   # TODO: migrate
    # src/opencl_wrapper/pipeline.cpp   # TODO: migrate
)

# Includes both 'include/' and 'src/' so header-only files can be included seamlessly
target_include_directories(opencl_wrapper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

target_link_libraries(opencl_wrapper PUBLIC OpenCL::OpenCL)

if(MSVC)
    target_compile_options(opencl_wrapper PRIVATE /W4)
else()
    target_compile_options(opencl_wrapper PRIVATE -Wall -Wextra)
endif()

target_compile_definitions(opencl_wrapper PUBLIC CL_TARGET_OPENCL_VERSION=300)

# ───────────────────────── examples (Layer 3) ────────────────────────────────
add_executable(example src/examples/example.cpp)
target_link_libraries(example PRIVATE opencl_wrapper)

# ───────────────────────── tests ─────────────────────────────────────────────
enable_testing()
set(WRAPPER_TESTS
    test_platform
    # test_buffer     # add once buffer.cpp is migrated
    # test_dispatch   # add once dispatch.cpp is migrated
    # test_pipeline   # add once pipeline.cpp is migrated
)

foreach(test_name ${WRAPPER_TESTS})
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE opencl_wrapper)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
