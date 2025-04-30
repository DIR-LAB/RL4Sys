cmake_minimum_required(VERSION 3.15)
project(rl4sys_cppclient VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Dependencies ---
# We'll need gRPC, Protobuf, and a JSON library.
# This example assumes they are findable by CMake (e.g., installed system-wide or via find_package).
# You might need to adjust this based on how you manage dependencies (Conan, vcpkg, submodules).

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(GTest REQUIRED) # Find GoogleTest
# Example using nlohmann/json (assuming it's available, perhaps as a submodule or fetched)
# add_subdirectory(third_party/json) # Or find_package if installed

# --- Protobuf Generation ---
# Assuming rl4sys.proto is in ../proto/
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
set(PROTO_FILE ${PROTO_DIR}/rl4sys.proto)
set(PROTO_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_proto)

# Ensure the output directory exists
file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

protobuf_generate(
    TARGET rl4sys_proto
    LANGUAGE cpp
    OUT_VAR PROTO_SRCS
    PLUGIN_OUT_VAR GRPC_SRCS
    PROTOS ${PROTO_FILE}
    IMPORT_DIRS ${PROTO_DIR}
    PLUGINS $<TARGET_FILE:grpc_cpp_plugin>
    GENERATE_EXTENSIONS .pb.h .pb.cc .grpc.pb.h .grpc.pb.cc
    OUTPUT_DIR ${PROTO_GENERATED_DIR}
)


# --- Library Definition ---
add_library(rl4sys_client STATIC
    src/rl4sys_agent.cpp
    src/config_loader.cpp
    # Add other source files as needed
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

target_include_directories(rl4sys_client PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROTO_GENERATED_DIR}> # Include generated proto headers
    # Add include path for nlohmann/json if needed
)

target_link_libraries(rl4sys_client PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
    gRPC::grpc
    # Link nlohmann_json::nlohmann_json if needed
)

# --- Testing --- 
enable_testing() # Enable CTest support

# Define the test executable
add_executable(cppclient_tests
    test/rl4sys_agent_test.cpp
    # Add other test source files here if needed
)

# Link the test executable against our library and GoogleTest
target_link_libraries(cppclient_tests PRIVATE
    rl4sys_client
    GTest::gtest      # Link GoogleTest library
    GTest::gtest_main # Link GoogleTest main function
)

# Include the source directory for finding headers during tests
# Use target_include_directories for better scoping
target_include_directories(cppclient_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROTO_GENERATED_DIR}
    # Add include path for nlohmann/json if needed
    # ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include
)

# Add the test to CTest
include(GoogleTest)
# Automatically discover tests in the executable
gtest_discover_tests(cppclient_tests
    # Set working directory if needed, e.g., to find test_conf.json
    # WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Example Executable (Optional) ---
# You might want an example executable to test the client
# add_executable(lunar_lander_cpp_client src/examples/lunar_lander_main.cpp)
# target_link_libraries(lunar_lander_cpp_client PRIVATE rl4sys_client)


# --- Installation (Optional) ---
# install(TARGETS rl4sys_client cppclient_tests EXPORT rl4sys_client_targets
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
#         RUNTIME DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)
# install(EXPORT rl4sys_client_targets
#         FILE rl4sys_client_targets.cmake
#         NAMESPACE rl4sys::
#         DESTINATION lib/cmake/rl4sys_client)
