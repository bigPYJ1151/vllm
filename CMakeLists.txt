cmake_minimum_required(VERSION 3.15)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")

project(PageAttCPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYTHON_LIB_PATH "/usr/local/lib/python3.8/dist-packages/")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PYTHON_LIB_PATH})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/FindModules")

option(USE_STATIC_MKL "Prefer to link with MKL statically (Unix only)" OFF)

add_subdirectory(csrc)