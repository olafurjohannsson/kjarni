# Speech to text in a Qt Quick app: record from the microphone or open a file, and
# transcribe it locally with Whisper.
#
#   cmake -B build -DCMAKE_PREFIX_PATH=/path/to/Qt/6.8.3/gcc_64 \
#                  -DKJARNI_LIB_DIR=/path/to/kjarni-x86_64-linux \
#                  -DKJARNI_INCLUDE_DIR=/path/to/kjarni-x86_64-linux
#   cmake --build build
#   ./build/kjarni_qt_speech
#
# KJARNI_LIB_DIR is the extracted release archive, or target/release in this repo.
# Qt Multimedia is a separate module: select it in the Qt installer.

cmake_minimum_required(VERSION 3.21)
project(kjarni_qt_speech LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)   # std::expected in kjarni.hpp
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Quick Concurrent Multimedia)
qt_standard_project_setup(REQUIRES 6.5)

set(KJARNI_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../target/release"
    CACHE PATH "Directory containing the Kjarni native library")
set(KJARNI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    CACHE PATH "Directory containing kjarni.h and kjarni.hpp")

qt_add_executable(kjarni_qt_speech main.cpp)

qt_add_qml_module(kjarni_qt_speech
    URI KjarniSpeech
    QML_FILES Main.qml
    SOURCES speechtotext.h speechtotext.cpp)

target_include_directories(kjarni_qt_speech PRIVATE "${KJARNI_INCLUDE_DIR}")
target_link_directories(kjarni_qt_speech PRIVATE "${KJARNI_LIB_DIR}")
target_link_libraries(kjarni_qt_speech PRIVATE
    Qt6::Quick Qt6::Concurrent Qt6::Multimedia kjarni_ffi)

# Find the library next to the build without LD_LIBRARY_PATH.
set_target_properties(kjarni_qt_speech PROPERTIES BUILD_RPATH "${KJARNI_LIB_DIR}")
