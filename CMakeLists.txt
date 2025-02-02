cmake_minimum_required(VERSION 3.22.1)
project(bits_and_bytes LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)

set(ERTS_INCLUDE_DIR "$ENV{ERTS_INCLUDE_DIR}")

add_library(bits_and_bytes SHARED 
    c_src/elixirInterface.cpp
    c_src/kernels.cu
    c_src/ops.cu 
)

target_include_directories(bits_and_bytes PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/c_src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${ERTS_INCLUDE_DIR}
)

target_link_libraries(bits_and_bytes PUBLIC 
    CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusparse
)

set_target_properties(bits_and_bytes PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/priv"
    PREFIX "lib"
    CUDA_SEPARABLE_COMPILATION ON
)