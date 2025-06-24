cmake_minimum_required(VERSION 3.20)
project(grpc_helloworld LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC     CONFIG REQUIRED)
find_package(absl     CONFIG  REQUIRED)

set(PROTO_FILES helloworld.proto)



add_library(helloworld ${PROTO_FILES})
target_link_libraries(helloworld
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc
        gRPC::grpc++
)
target_include_directories(helloworld PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate(TARGET helloworld LANGUAGE cpp)
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET helloworld LANGUAGE grpc 
GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc 
PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")


add_executable(server server.cpp)
target_link_libraries(server PRIVATE helloworld
    )


add_executable(client client.cpp)
target_link_libraries(client PRIVATE helloworld)
