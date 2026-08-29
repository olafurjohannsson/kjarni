# Kjarni in a Qt Quick application.
#
#   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/Qt/6.x/gcc_64 \
#                  -DKJARNI_LIB_DIR=/path/to/target/release
#   cmake --build build
#   ./build/kjarni-qml
#
# Qt 6.5+ for QML_ELEMENT auto-registration and loadFromModule.

cmake_minimum_required(VERSION 3.21)
project(kjarni_qml VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Quick Concurrent)
qt_standard_project_setup(REQUIRES 6.5)

set(KJARNI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../target/release"
    CACHE PATH "Directory containing libkjarni_ffi")

qt_add_executable(kjarni-qml main.cpp)

# KjarniModel.h is listed as a source so AUTOMOC processes its Q_OBJECT classes and
# the QML_ELEMENT registrations reach the type system.
qt_add_qml_module(kjarni-qml
    URI Kjarni
    VERSION 1.0
    QML_FILES Main.qml
    SOURCES KjarniModel.h
)

target_include_directories(kjarni-qml PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_directories(kjarni-qml PRIVATE "${KJARNI_LIB_DIR}")
target_link_libraries(kjarni-qml PRIVATE Qt6::Quick Qt6::Concurrent kjarni_ffi)

set_target_properties(kjarni-qml PROPERTIES
    BUILD_RPATH "${KJARNI_LIB_DIR}"
    INSTALL_RPATH "${KJARNI_LIB_DIR}")
