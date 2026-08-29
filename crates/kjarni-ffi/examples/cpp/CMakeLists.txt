# Minimal CMake setup for using Kjarni from C++.
#
#   cmake -B build -DKJARNI_LIB_DIR=/path/to/target/release
#   cmake --build build
#   LD_LIBRARY_PATH=$KJARNI_LIB_DIR ./build/example
#
# Qt, or any other C++ project, needs nothing more than this: one header and one
# shared library.

cmake_minimum_required(VERSION 3.16)
project(kjarni_cpp_example CXX)

set(CMAKE_CXX_STANDARD 23)   # std::expected in kjarni.hpp
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Where libkjarni_ffi.{so,dylib,dll} lives. Defaults to a debug/release build in
# this repo; point it at the extracted release archive otherwise.
set(KJARNI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../target/release"
    CACHE PATH "Directory containing the Kjarni native library")

add_executable(example example.cpp)

# GCC warns about omitted fields in C++20 designated initialisers, which is exactly
# how kjarni.hpp's option structs are meant to be used.
target_compile_options(example PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-missing-field-initializers>)

target_include_directories(example PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_directories(example PRIVATE "${KJARNI_LIB_DIR}")
target_link_libraries(example PRIVATE kjarni_ffi)

# So the binary finds the library without LD_LIBRARY_PATH after installation.
set_target_properties(example PROPERTIES
    BUILD_RPATH "${KJARNI_LIB_DIR}"
    INSTALL_RPATH "${KJARNI_LIB_DIR}")
