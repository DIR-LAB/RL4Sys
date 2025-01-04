cmake_minimum_required(VERSION 3.10)
project(ClientLib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find gRPC and Protocol Buffers
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/generated
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/rlsys_lib.cpp
    src/rlsys_handler.cpp
    src/main.cpp
    generated/model_service.pb.cc
    generated/model_service.grpc.pb.cc
)

# Create the executable or library
add_executable(client_app ${SOURCES})

# Link libraries
target_link_libraries(client_app
    ${Protobuf_LIBRARIES}
    ${gRPC_LIBRARIES}
)

# Set properties
set_property(TARGET client_app PROPERTY CXX_STANDARD 14)