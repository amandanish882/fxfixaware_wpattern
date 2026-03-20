cmake_minimum_required(VERSION 3.15)
project(fx_pricing_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Find pybind11 ────────────────────────────────────────────────
# Try system install first, then fall back to pip-installed pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
    else()
        # Last resort: use pip-installed pybind11
        execute_process(
            COMMAND python -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
    endif()
endif()

# ── Include directories ──────────────────────────────────────────
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Python module ────────────────────────────────────────────────
pybind11_add_module(fx_pricing_kernel
    bindings/pybind_module.cpp
)

target_compile_definitions(fx_pricing_kernel PRIVATE
    NDEBUG
)

# MSVC-specific settings
if(MSVC)
    target_compile_options(fx_pricing_kernel PRIVATE
        /O2 /EHsc /W3
    )
else()
    target_compile_options(fx_pricing_kernel PRIVATE
        -O3 -Wall -Wextra
    )
endif()

# ── Install: copy .pyd/.so to project root ───────────────────────
set(FX_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
install(TARGETS fx_pricing_kernel
    LIBRARY DESTINATION ${FX_PROJECT_ROOT}
    RUNTIME DESTINATION ${FX_PROJECT_ROOT}
)

# ── Print build info ─────────────────────────────────────────────
message(STATUS "fx_pricing_kernel: C++${CMAKE_CXX_STANDARD}")
message(STATUS "fx_pricing_kernel: pybind11 ${pybind11_VERSION}")
message(STATUS "fx_pricing_kernel: install -> ${FX_PROJECT_ROOT}")
