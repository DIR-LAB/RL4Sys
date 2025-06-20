cmake_minimum_required(VERSION 3.15)
project(rl4sys_cppclient VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# --- Dependencies ---
find_package(PkgConfig REQUIRED)

# --- PyTorch C++ Support (find first to avoid conflicts) ---
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "PyTorch found: ${TORCH_LIBRARIES}")
    set(USE_PYTORCH ON)
    add_definitions(-DUSE_PYTORCH)
else()
    message(WARNING "PyTorch not found - will use random model fallback only")
    set(USE_PYTORCH OFF)
endif()

# Special handling for macOS Homebrew installations to avoid target conflicts
if(APPLE AND NOT USE_PYTORCH)
    # Use pkg-config approach when PyTorch is not available to avoid conflicts
    pkg_check_modules(PROTOBUF REQUIRED protobuf)
    pkg_check_modules(GRPC REQUIRED grpc++)
    pkg_check_modules(ABSL REQUIRED absl_base absl_strings absl_cord)
    
    # Set up protobuf and gRPC variables using pkg-config results
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
    
    # Create interface targets to maintain compatibility
    if(NOT TARGET protobuf::libprotobuf)
        add_library(protobuf::libprotobuf INTERFACE IMPORTED)
        target_link_libraries(protobuf::libprotobuf INTERFACE ${PROTOBUF_LIBRARIES})
        target_include_directories(protobuf::libprotobuf INTERFACE ${PROTOBUF_INCLUDE_DIRS})
        target_link_directories(protobuf::libprotobuf INTERFACE ${PROTOBUF_LIBRARY_DIRS})
    endif()
    
    if(NOT TARGET gRPC::grpc++)
        add_library(gRPC::grpc++ INTERFACE IMPORTED)
        target_link_libraries(gRPC::grpc++ INTERFACE ${GRPC_LIBRARIES})
        target_include_directories(gRPC::grpc++ INTERFACE ${GRPC_INCLUDE_DIRS})
        target_link_directories(gRPC::grpc++ INTERFACE ${GRPC_LIBRARY_DIRS})
    endif()
    
    if(NOT TARGET gRPC::grpc)
        add_library(gRPC::grpc INTERFACE IMPORTED)
        target_link_libraries(gRPC::grpc INTERFACE ${GRPC_LIBRARIES})
        target_include_directories(gRPC::grpc INTERFACE ${GRPC_INCLUDE_DIRS})
        target_link_directories(gRPC::grpc INTERFACE ${GRPC_LIBRARY_DIRS})
    endif()
elseif(NOT APPLE)
    # For non-macOS systems, use the standard CMake approach
    find_package(Protobuf REQUIRED)
    find_package(gRPC REQUIRED)
endif()

# If PyTorch is found on macOS, protobuf is available via PyTorch
# but we still need to find gRPC separately
if(APPLE AND USE_PYTORCH)
    pkg_check_modules(GRPC REQUIRED grpc++)
    
    # Create gRPC targets
    if(NOT TARGET gRPC::grpc++)
        add_library(gRPC::grpc++ INTERFACE IMPORTED)
        target_link_libraries(gRPC::grpc++ INTERFACE ${GRPC_LIBRARIES})
        target_include_directories(gRPC::grpc++ INTERFACE ${GRPC_INCLUDE_DIRS})
        target_link_directories(gRPC::grpc++ INTERFACE ${GRPC_LIBRARY_DIRS})
    endif()
    
    if(NOT TARGET gRPC::grpc)
        add_library(gRPC::grpc INTERFACE IMPORTED)
        target_link_libraries(gRPC::grpc INTERFACE ${GRPC_LIBRARIES})
        target_include_directories(gRPC::grpc INTERFACE ${GRPC_INCLUDE_DIRS})
        target_link_directories(gRPC::grpc INTERFACE ${GRPC_LIBRARY_DIRS})
    endif()
endif()

# Find nlohmann_json
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    # Try to find it via pkg-config
    pkg_check_modules(nlohmann_json REQUIRED nlohmann_json)
endif()

# Find GoogleTest for testing
find_package(GTest REQUIRED)

# Find zlib for model decompression
find_package(ZLIB REQUIRED)

# --- Protobuf Generation ---
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
set(PROTO_FILE ${PROTO_DIR}/rl4sys.proto)
set(PROTO_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_proto)

# Ensure the output directory exists
file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Generate protobuf files
set(PROTO_SRCS
    ${PROTO_GENERATED_DIR}/rl4sys.pb.cc
)
set(PROTO_HDRS  
    ${PROTO_GENERATED_DIR}/rl4sys.pb.h
)
set(GRPC_SRCS
    ${PROTO_GENERATED_DIR}/rl4sys.grpc.pb.cc
)
set(GRPC_HDRS
    ${PROTO_GENERATED_DIR}/rl4sys.grpc.pb.h
)

# Find required programs (moved outside the APPLE-specific section)
if(NOT PROTOBUF_PROTOC)
    find_program(PROTOBUF_PROTOC protoc REQUIRED)
endif()
if(NOT GRPC_CPP_PLUGIN)
    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
endif()

# Custom command to generate protobuf files
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOBUF_PROTOC}
    ARGS --grpc_out=${PROTO_GENERATED_DIR}
         --cpp_out=${PROTO_GENERATED_DIR}
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         --proto_path=${PROTO_DIR}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf and gRPC files"
    VERBATIM
)

# Create a custom target for generated files
add_custom_target(generate_proto_files DEPENDS ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS})

# --- Library Definition ---
add_library(rl4sys_client STATIC
    # Core implementation files
    src/rl4sys_types.cpp
    src/rl4sys_agent.cpp
    src/config_loader.cpp
    src/logger.cpp
    src/model_manager.cpp
    src/rl4sys_stub_wrapper.cpp
    
    # Utility files
    src/utils/compression.cpp
    
    # PyTorch model files (conditional)
    $<$<BOOL:${USE_PYTORCH}>:src/pytorch_model_wrapper.cpp>
    $<$<BOOL:${USE_PYTORCH}>:src/ppo_model.cpp>
    $<$<BOOL:${USE_PYTORCH}>:src/dqn_model.cpp>
    
    # Generated protobuf files
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# Make sure protobuf files are generated before building the library
add_dependencies(rl4sys_client generate_proto_files)

# Include directories
target_include_directories(rl4sys_client PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROTO_GENERATED_DIR}>
)

# Link libraries
target_link_libraries(rl4sys_client PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
    gRPC::grpc
    ZLIB::ZLIB
    $<$<BOOL:${USE_PYTORCH}>:${TORCH_LIBRARIES}>
)

# Handle nlohmann_json linking
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(rl4sys_client PUBLIC nlohmann_json::nlohmann_json)
elseif(nlohmann_json_FOUND)
    target_include_directories(rl4sys_client PUBLIC ${nlohmann_json_INCLUDE_DIRS})
    target_link_libraries(rl4sys_client PUBLIC ${nlohmann_json_LIBRARIES})
endif()

# Compiler-specific options
target_compile_options(rl4sys_client PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# --- Testing ---
enable_testing()

# Define the test executable
add_executable(cppclient_tests
    test/rl4sys_agent_test.cpp
)

# Make sure protobuf files are generated before building tests
add_dependencies(cppclient_tests generate_proto_files)

# Link the test executable
target_link_libraries(cppclient_tests PRIVATE
    rl4sys_client
    GTest::gtest
    GTest::gtest_main
)

# Include directories for tests
target_include_directories(cppclient_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROTO_GENERATED_DIR}
)

# Add the test to CTest
include(GoogleTest)
gtest_discover_tests(cppclient_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Example Executable (Optional) ---
add_executable(lunar_lander_cpp_example
    examples/lunar_lander_cpp_main.cpp
)

target_link_libraries(lunar_lander_cpp_example PRIVATE 
    rl4sys_client
)

target_include_directories(lunar_lander_cpp_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROTO_GENERATED_DIR}
)

# Make sure protobuf files are generated before building example
add_dependencies(lunar_lander_cpp_example generate_proto_files)

# --- Installation (Optional) ---
install(TARGETS rl4sys_client cppclient_tests lunar_lander_cpp_example
        EXPORT rl4sys_client_targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION include)

install(FILES ${PROTO_HDRS} ${GRPC_HDRS} DESTINATION include)

install(EXPORT rl4sys_client_targets
        FILE rl4sys_client_targets.cmake
        NAMESPACE rl4sys::
        DESTINATION lib/cmake/rl4sys_client)

# --- Print Configuration Summary ---
message(STATUS "=== RL4Sys C++ Client Configuration ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(APPLE)
    message(STATUS "Protobuf Libraries: ${PROTOBUF_LIBRARIES}")
    message(STATUS "gRPC Libraries: ${GRPC_LIBRARIES}")
    message(STATUS "Abseil Libraries: ${ABSL_LIBRARIES}")
else()
    message(STATUS "Protobuf Version: ${Protobuf_VERSION}")
    message(STATUS "gRPC Found: ${gRPC_FOUND}")
endif()
message(STATUS "GTest Found: ${GTest_FOUND}")
if(TARGET nlohmann_json::nlohmann_json)
    message(STATUS "nlohmann_json: Found (target)")
elseif(nlohmann_json_FOUND)
    message(STATUS "nlohmann_json: Found (pkg-config)")
else()
    message(WARNING "nlohmann_json: Not found - build may fail")
endif()
message(STATUS "PyTorch Support: ${USE_PYTORCH}")
if(USE_PYTORCH)
    message(STATUS "PyTorch Libraries: ${TORCH_LIBRARIES}")
endif()
message(STATUS "zlib Found: ${ZLIB_FOUND}")
message(STATUS "Proto Files: ${PROTO_FILE}")
message(STATUS "Generated Proto Dir: ${PROTO_GENERATED_DIR}")
message(STATUS "=======================================")

# --- Helper Targets ---
# Target to clean generated files
add_custom_target(clean_proto
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${PROTO_GENERATED_DIR}
    COMMENT "Cleaning generated protobuf files"
)

# Target to regenerate proto files
add_custom_target(regen_proto
    DEPENDS clean_proto generate_proto_files
    COMMENT "Regenerating protobuf files"
)