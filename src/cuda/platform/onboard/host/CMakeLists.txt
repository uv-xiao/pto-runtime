# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)
project(cuda_host_runtime LANGUAGES C CXX)

find_package(CUDAToolkit REQUIRED)

set(CUDA_HOST_RUNTIME_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/pto_runtime_c_api.cpp"
)

if(DEFINED CUSTOM_SOURCE_DIRS)
    foreach(SRC_DIR ${CUSTOM_SOURCE_DIRS})
        file(GLOB_RECURSE DIR_SOURCES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c")
        list(APPEND CUDA_HOST_RUNTIME_SOURCES ${DIR_SOURCES})
    endforeach()
endif()

add_library(host_runtime SHARED ${CUDA_HOST_RUNTIME_SOURCES})

set_target_properties(host_runtime PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "host_runtime"
    SUFFIX ".so"
)

target_compile_options(host_runtime
    PRIVATE
        -Wall
        -Wextra
        -fPIC
        -O3
        -g
        $<$<CXX_COMPILER_ID:GNU>:-fno-gnu-unique>
)

target_include_directories(host_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/worker
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common/platform_comm
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
)

if(DEFINED CUSTOM_INCLUDE_DIRS)
    target_include_directories(host_runtime PRIVATE ${CUSTOM_INCLUDE_DIRS})
endif()

target_link_libraries(host_runtime
    PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        dl
)
