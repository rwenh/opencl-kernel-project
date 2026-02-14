cmake_minimum_required(VERSION 3.16)
project(MyOpenCLProject VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

# compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g") # set debugging flags

# collect all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS}
        src/opencl_wrapper/buffer.cpp)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        OpenCL::OpenCL
)

# make include directory available to all source files
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)