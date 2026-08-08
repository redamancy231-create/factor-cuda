# factor-cuda — CMake 构建骨架
# Phase 0 自检 target + 后续 src/ 扩展点（见 PLAN.md §六 模块拆分）
cmake_minimum_required(VERSION 3.22)
project(factor_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 13.x CCCL (libcu++/CUB) requires the conformant MSVC preprocessor;
# the traditional preprocessor is a fatal error (preprocessor.h:20).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")

# CUDA 目标架构：RTX 4060 Laptop (sm_89, Ada)
# CMake 对无后缀架构自动生成 real(SASS) + virtual(PTX 前向 JIT)，无需重复写 89-virtual
# 多架构支持矩阵（CUDA/MSVC/Python/compute capability）为 P1 待办，定义后从不重复列表导出
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING
    "CUDA target architecture: sm_89 (auto real+virtual)")

# ---- Phase 0 自检：工具链验证（GPU vec_add）----
add_executable(phase0_selfcheck poc/phase0_selfcheck.cu)
set_target_properties(phase0_selfcheck PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# ---- PoC 3：cs_rank v0 kernel ----
# CUB ships under <cuda>/include/cccl in CUDA 13.x. Resolve via the CUDA_PATH
# env var (set by the CUDA Toolkit installer) so no machine-specific path is
# hard-coded.
find_path(CUB_CCCL_INCLUDE_DIR
    NAMES cub/device/device_segmented_radix_sort.cuh
    PATHS "$ENV{CUDA_PATH}/include/cccl"
    NO_DEFAULT_PATH)
if(CUB_CCCL_INCLUDE_DIR)
    message(STATUS "CUB cccl include dir: ${CUB_CCCL_INCLUDE_DIR}")
endif()

# correctness selfcheck (parity anchors + randomized bitwise vs CPU reference)
add_executable(poc3_cs_rank_selfcheck
    poc/poc3_cs_rank_selfcheck.cu
    src/cross_sectional_rank.cu
    src/mem_tracker.cu)
set_target_properties(poc3_cs_rank_selfcheck PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_cs_rank_selfcheck PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_cs_rank_selfcheck PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# perf + memory probe (corpus scale 1218x5000, gate 13.926ms, peak sample)
add_executable(poc3_cs_rank_perf
    poc/poc3_cs_rank_perf.cu
    src/cross_sectional_rank.cu
    src/mem_tracker.cu)
set_target_properties(poc3_cs_rank_perf PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_cs_rank_perf PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_cs_rank_perf PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# mem_tracker three-way calibration (theory vs HWM vs driver sample)
add_executable(poc3_mem_tracker_selfcheck
    poc/poc3_mem_tracker_selfcheck.cu
    src/cross_sectional_rank.cu
    src/mem_tracker.cu)
set_target_properties(poc3_mem_tracker_selfcheck PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_mem_tracker_selfcheck PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_mem_tracker_selfcheck PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# rolling_ic v0 selfcheck (parity anchors + randomized vs CPU reference)
add_executable(poc3_rolling_ic_selfcheck
    poc/poc3_rolling_ic_selfcheck.cu
    src/rolling_ic.cu
    src/mem_tracker.cu)
set_target_properties(poc3_rolling_ic_selfcheck PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_rolling_ic_selfcheck PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_rolling_ic_selfcheck PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# rolling_ic v0 perf + memory probe (corpus scale, gate 77.18ms)
add_executable(poc3_rolling_ic_perf
    poc/poc3_rolling_ic_perf.cu
    src/rolling_ic.cu
    src/mem_tracker.cu)
set_target_properties(poc3_rolling_ic_perf PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_rolling_ic_perf PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_rolling_ic_perf PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# factor_corr v0 selfcheck (corpus anchors + randomized vs CPU reference)
add_executable(poc3_factor_corr_selfcheck
    poc/poc3_factor_corr_selfcheck.cu
    src/factor_corr.cu
    src/mem_tracker.cu)
set_target_properties(poc3_factor_corr_selfcheck PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_factor_corr_selfcheck PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_factor_corr_selfcheck PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# factor_corr v0 perf + memory probe (corpus scale, gate 1543.52ms)
add_executable(poc3_factor_corr_perf
    poc/poc3_factor_corr_perf.cu
    src/factor_corr.cu
    src/mem_tracker.cu)
set_target_properties(poc3_factor_corr_perf PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_factor_corr_perf PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_factor_corr_perf PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# stock_corr v0 selfcheck (parity anchors + randomized vs CPU reference)
add_executable(poc3_stock_corr_selfcheck
    poc/poc3_stock_corr_selfcheck.cu
    src/stock_corr.cu
    src/mem_tracker.cu)
set_target_properties(poc3_stock_corr_selfcheck PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_stock_corr_selfcheck PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_stock_corr_selfcheck PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# stock_corr v0 perf + memory probe (corpus scale, gate 21.8ms)
add_executable(poc3_stock_corr_perf
    poc/poc3_stock_corr_perf.cu
    src/stock_corr.cu
    src/mem_tracker.cu)
set_target_properties(poc3_stock_corr_perf PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_stock_corr_perf PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_stock_corr_perf PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# parameter_scan v0 selfcheck (dict order anchors + vs-single + randomized vs CPU)
add_executable(poc3_parameter_scan_selfcheck
    poc/poc3_parameter_scan_selfcheck.cu
    src/cross_sectional_rank.cu
    src/mem_tracker.cu)
set_target_properties(poc3_parameter_scan_selfcheck PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_parameter_scan_selfcheck PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_parameter_scan_selfcheck PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# parameter_scan v0 perf + memory probe (corpus scale, gate 55.34ms)
add_executable(poc3_parameter_scan_perf
    poc/poc3_parameter_scan_perf.cu
    src/cross_sectional_rank.cu
    src/mem_tracker.cu)
set_target_properties(poc3_parameter_scan_perf PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_parameter_scan_perf PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_parameter_scan_perf PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# five-op three-way memory calibration (theory vs HWM vs driver sample)
add_executable(poc3_calibration
    poc/poc3_calibration.cu
    src/cross_sectional_rank.cu
    src/rolling_ic.cu
    src/factor_corr.cu
    src/stock_corr.cu
    src/mem_tracker.cu)
set_target_properties(poc3_calibration PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_calibration PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_calibration PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# frozen-corpus cross-end parity runner (GPU side; Python side compares vs
# corr_oracle_v1.py)
add_executable(poc3_corpus_parity
    poc/poc3_corpus_parity.cu
    src/factor_corr.cu
    src/stock_corr.cu
    src/mem_tracker.cu)
set_target_properties(poc3_corpus_parity PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
target_include_directories(poc3_corpus_parity PRIVATE src)
if(CUB_CCCL_INCLUDE_DIR)
    target_include_directories(poc3_corpus_parity PRIVATE "${CUB_CCCL_INCLUDE_DIR}")
endif()

# ---- optional pybind11 module: factor_corr (GIL release + NumPy paths) ----
# Default OFF so the plain CUDA selfcheck/perf build is unaffected. Enable with
#   cmake -S . -B build -DBUILD_PYBIND11=ON -DPython_EXECUTABLE=<python3.12.exe>
# The binding source (src/factor_corr_pybind.cpp) only adds a thin wrapper; it
# does not touch the production kernels (factor_corr.cu / factor_corr_impl.cuh),
# keeping the F/T minimal-proof-2 review material stable.
option(BUILD_PYBIND11 "Build the factor_corr pybind11 module" OFF)
if(BUILD_PYBIND11)
    set(Python_EXECUTABLE "" CACHE FILEPATH "Path to python.exe")
    if(Python_EXECUTABLE)
        get_filename_component(Python_ROOT_DIR "${Python_EXECUTABLE}" DIRECTORY)
    endif()
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    set(pybind11_DIR "" CACHE PATH "pybind11 cmake config dir")
    if(NOT pybind11_DIR)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE _pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_pybind11_cmake_dir)
            set(pybind11_DIR "${_pybind11_cmake_dir}")
        endif()
    endif()
    find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}")
    # CXX sources (the .cpp binding) need the CUDA toolkit headers explicitly;
    # nvcc adds them automatically for .cu sources but cl.exe does not.
    find_package(CUDAToolkit REQUIRED)
    pybind11_add_module(factor_corr_pybind MODULE src/factor_corr_pybind.cpp src/factor_corr.cu src/mem_tracker.cu)
    set_target_properties(factor_corr_pybind PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_include_directories(factor_corr_pybind PRIVATE src ${CUDAToolkit_INCLUDE_DIRS})

    # four-kernel binding (cs_rank / rolling_ic / stock_corr / parameter_scan).
    # cross_sectional_rank.cu defines BOTH cs_rank_gpu and parameter_scan_gpu.
    pybind11_add_module(factor_cuda_pybind MODULE
        src/factor_cuda_pybind.cpp
        src/cross_sectional_rank.cu
        src/rolling_ic.cu
        src/stock_corr.cu
        src/mem_tracker.cu)
    set_target_properties(factor_cuda_pybind PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_include_directories(factor_cuda_pybind PRIVATE src ${CUDAToolkit_INCLUDE_DIRS})
endif()
