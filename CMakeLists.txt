cmake_minimum_required(VERSION 3.28)
project(gpu_note LANGUAGES CXX CUDA)

# 可选：设置 CUDA 相关编译选项
set(CMAKE_BUILD_TYPE release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)   # 支持 device code 分离编译

add_definitions(-DFMT_CONSTEVAL=)

find_package(fmt REQUIRED)
find_package(folly REQUIRED)
find_package(Boost REQUIRED)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/src/gemm/*.cu"
)

set(SOURCES ${SOURCE_FILES})

add_executable(${PROJECT_NAME} ${SOURCES})

find_package(CUDAToolkit REQUIRED)


set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES native
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::curand)
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
target_link_libraries(${PROJECT_NAME} PRIVATE folly glog gflags Boost::context double-conversion event)

