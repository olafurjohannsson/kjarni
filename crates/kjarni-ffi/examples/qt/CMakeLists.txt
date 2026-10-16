# Search a help centre from a Qt Quick app: keyword (BM25), semantic or hybrid
# retrieval as you type, with cross-encoder reranking on request.
#
#   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/Qt/6.8.3/gcc_64 \
#                  -DKJARNI_LIB_DIR=/path/to/kjarni-x86_64-linux
#   cmake --build build
#   ./build/kjarni_qt_search
#
# KJARNI_LIB_DIR is the extracted release archive, or target/release in this repo.

cmake_minimum_required(VERSION 3.21)
project(kjarni_qt_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)   # std::expected in kjarni.hpp
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Quick Concurrent)
qt_standard_project_setup(REQUIRES 6.5)

set(KJARNI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../target/release"
    CACHE PATH "Directory containing the Kjarni native library")
set(KJARNI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    CACHE PATH "Directory containing kjarni.h and kjarni.hpp")

qt_add_executable(kjarni_qt_search main.cpp)

qt_add_qml_module(kjarni_qt_search
    URI KjarniSearch
    QML_FILES Main.qml
    SOURCES helpsearch.h helpsearch.cpp articles.h)

target_include_directories(kjarni_qt_search PRIVATE "${KJARNI_INCLUDE_DIR}")
target_link_directories(kjarni_qt_search PRIVATE "${KJARNI_LIB_DIR}")
target_link_libraries(kjarni_qt_search PRIVATE Qt6::Quick Qt6::Concurrent kjarni_ffi)

# Find the library next to the build without LD_LIBRARY_PATH.
set_target_properties(kjarni_qt_search PROPERTIES BUILD_RPATH "${KJARNI_LIB_DIR}")
