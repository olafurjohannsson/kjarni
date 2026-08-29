# C++ wrapper tests.
#
#   cmake -B build -DKJARNI_LIB_DIR=/path/to/target/release
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Add -DKJARNI_SANITIZE=ON to run under AddressSanitizer and UBSan, which is what
# makes the ownership assertions meaningful.

cmake_minimum_required(VERSION 3.16)
project(kjarni_cpp_tests CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KJARNI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../target/release"
    CACHE PATH "Directory containing the Kjarni native library")
option(KJARNI_SANITIZE "Build with AddressSanitizer and UBSan" OFF)

enable_testing()

add_executable(test_kjarni test_kjarni.cpp)

target_include_directories(test_kjarni PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_directories(test_kjarni PRIVATE "${KJARNI_LIB_DIR}")
target_link_libraries(test_kjarni PRIVATE kjarni_ffi)

# The option structs are designed for designated initialisers; GCC's -Wextra
# complains about every omitted field.
target_compile_options(test_kjarni PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wno-missing-field-initializers>)

if(KJARNI_SANITIZE)
    target_compile_options(test_kjarni PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(test_kjarni PRIVATE -fsanitize=address,undefined)
endif()

set_target_properties(test_kjarni PROPERTIES BUILD_RPATH "${KJARNI_LIB_DIR}")

# Models download on first use, so the first run can be slow.
add_test(NAME cpp_wrapper COMMAND test_kjarni)
set_tests_properties(cpp_wrapper PROPERTIES TIMEOUT 600)
