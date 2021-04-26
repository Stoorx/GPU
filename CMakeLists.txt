cmake_minimum_required(VERSION 3.17)
project(GPU)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(lib/OpenCL-SDK)
include_directories(lib/OpenCL-SDK/external/OpenCL-CLHPP/include)
include_directories(lib/OpenCL-SDK/external/OpenCL-Headers)
add_executable(GPU main.cpp Application.cpp Application.h)
target_link_libraries(GPU PUBLIC ${CMAKE_BINARY_DIR}/lib/OpenCL-SDK/external/OpenCL-ICD-Loader/OpenCl.lib)