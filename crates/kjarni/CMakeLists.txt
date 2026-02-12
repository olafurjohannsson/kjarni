cmake_minimum_required(VERSION 3.15)
project(edgegpt_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Rust library
set(EDGEGPT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link directories
link_directories(${EDGEGPT_LIB_DIR})

# Example executable
add_executable(edgegpt_example examples/example.cpp)

# Link against the Rust library
if(WIN32)
    target_link_libraries(edgegpt_example edgegpt.dll.lib ws2_32 userenv bcrypt ntdll)
elseif(APPLE)
    target_link_libraries(edgegpt_example edgegpt "-framework Security" "-framework CoreFoundation" pthread dl m)
else()
    target_link_libraries(edgegpt_example edgegpt pthread dl m)
endif()

# Set runtime library search path
if(APPLE)
    set_target_properties(edgegpt_example PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(edgegpt_example PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()