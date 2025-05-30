cmake_minimum_required(VERSION 3.15)
project(rl4sys_test_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Manually add CUDA
set(CMAKE_CUDA_ARCHITECTURES native)
# set(CMAKE_CUDA_ARCHITECTURES 60 75 86)

# --- Dependencies ---
list(APPEND CMAKE_PREFIX_PATH "/home/yomi/libtorch")

find_package(Torch REQUIRED)                    # Export target: torch::torch
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(PkgConfig REQUIRED)

# Find nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # Try to find it via pkg-config
    pkg_check_modules(NLOHMANN_JSON REQUIRED nlohmann_json)
endif()

# Find spdlog
find_package(spdlog REQUIRED)

# --- Protobuf Generation ---
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../proto)
set(PROTO_FILE ${PROTO_DIR}/rl4sys.proto)
set(PROTO_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_proto)

# Ensure the output directory exists
file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Set the generated files paths
set(PROTO_SRCS "${PROTO_GENERATED_DIR}/rl4sys.pb.cc")
set(PROTO_HDRS "${PROTO_GENERATED_DIR}/rl4sys.pb.h")
set(GRPC_SRCS "${PROTO_GENERATED_DIR}/rl4sys.grpc.pb.cc")
set(GRPC_HDRS "${PROTO_GENERATED_DIR}/rl4sys.grpc.pb.h")

# Get the gRPC plugin location
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)

# Custom command for protobuf and gRPC generation
add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out=${PROTO_GENERATED_DIR}
         --grpc_out=${PROTO_GENERATED_DIR}
         --plugin=protoc-gen-grpc=${grpc_cpp_plugin_location}
         -I${PROTO_DIR}
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf and gRPC files"
    VERBATIM
)

# --- Executable ---
add_executable(test_cpp_client 
    test.cpp
    ../../cppclient/src/rl4sys_agent.cpp
    ../../cppclient/src/rl4sys_types.cpp
    ../../cppclient/src/config_loader.cpp
    ../../cppclient/src/util.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

target_include_directories(test_cpp_client PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cppclient/include
    ${PROTO_GENERATED_DIR}
)

target_link_libraries(test_cpp_client PRIVATE
    ${TORCH_LIBRARIES}
    gRPC::grpc++
    gRPC::grpc
    spdlog::spdlog
)

# Use pkg-config for protobuf to handle all dependencies
pkg_check_modules(PROTOBUF REQUIRED protobuf)
target_link_libraries(test_cpp_client PRIVATE ${PROTOBUF_LIBRARIES})
target_include_directories(test_cpp_client PRIVATE ${PROTOBUF_INCLUDE_DIRS})

# Link nlohmann/json
if(nlohmann_json_FOUND)
    target_link_libraries(test_cpp_client PRIVATE nlohmann_json::nlohmann_json)
else()
    target_include_directories(test_cpp_client PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
    target_link_libraries(test_cpp_client PRIVATE ${NLOHMANN_JSON_LIBRARIES})
endif()

# Set C++ properties for PyTorch
set_property(TARGET test_cpp_client PROPERTY CXX_STANDARD 17)

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
