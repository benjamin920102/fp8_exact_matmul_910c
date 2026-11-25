cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture")
find_package(ASC REQUIRED)
project(fp8_exact_matmul LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ASC_STANDARD 17)
add_executable(demo fp8_matmul.asc main.cpp)
target_include_directories(demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(DEFINED ENV{ASCEND_HOME_PATH})
    target_include_directories(demo PRIVATE "$ENV{ASCEND_HOME_PATH}/include")
    target_link_directories(demo PRIVATE "$ENV{ASCEND_HOME_PATH}/lib64")
endif()
target_compile_options(demo PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
# ASC generated host stubs need the actual Ascend C runtime, not just ACL.
# Put the static archive before its dependencies and keep shared dependencies
# through the link. A group also resolves cyclic archive references.
find_library(ASCENDC_RUNTIME_LIB NAMES ascendc_runtime
    HINTS "$ENV{ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
if(NOT ASCENDC_RUNTIME_LIB)
    message(FATAL_ERROR "ascendc_runtime not found in $ENV{ASCEND_HOME_PATH}/lib64; check CANN_ROOT")
endif()
message(STATUS "ASC host runtime: ${ASCENDC_RUNTIME_LIB}")

set(ASC_EXTRA_LIBS "")
foreach(lib runtime error_manager ge_common_base graph_base acl_rt)
    find_library(ASC_OPTIONAL_${lib} NAMES ${lib}
        HINTS "$ENV{ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
    if(ASC_OPTIONAL_${lib})
        list(APPEND ASC_EXTRA_LIBS "${ASC_OPTIONAL_${lib}}")
    endif()
endforeach()
find_package(Threads REQUIRED)
target_link_libraries(demo PRIVATE
    "-Wl,--push-state,--no-as-needed"
    "-Wl,--start-group"
    "${ASCENDC_RUNTIME_LIB}"
    ascendcl tiling_api register platform unified_dlog c_sec mmpa profapi
    ${ASC_EXTRA_LIBS}
    Threads::Threads ${CMAKE_DL_LIBS} m
    "-Wl,--end-group"
    "-Wl,--pop-state"
)
