cmake_minimum_required(VERSION 3.15)
project(edgegpt_cpp_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Rust library location
set(EDGEGPT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../target/release")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
link_directories(${EDGEGPT_LIB_DIR})

# Helper function to create examples
function(add_edgegpt_example name)
    add_executable(${name} ${name}.cpp)
    
    if(WIN32)
        target_link_libraries(${name} edgegpt ws2_32 userenv bcrypt ntdll)
    elseif(APPLE)
        target_link_libraries(${name} edgegpt "-framework Security" "-framework CoreFoundation" pthread dl m)
    else()
        # Linux - link with absolute path
        target_link_libraries(${name} ${EDGEGPT_LIB_DIR}/libedgegpt.so pthread dl m)
    endif()
    
    # Set RPATH to find libedgegpt.so at runtime
    if(UNIX)
        set_target_properties(${name} PROPERTIES
            BUILD_RPATH "${EDGEGPT_LIB_DIR}"
            INSTALL_RPATH "${EDGEGPT_LIB_DIR}"
        )
    endif()
endfunction()

# Add all examples
add_edgegpt_example(01_sentence_encoding)
add_edgegpt_example(02_batch_encoding)
add_edgegpt_example(03_similarity)
add_edgegpt_example(04_semantic_search)
add_edgegpt_example(05_reranking)