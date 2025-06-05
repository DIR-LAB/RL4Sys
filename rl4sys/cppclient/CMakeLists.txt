cmake_minimum_required(VERSION 3.15)
project(rl4sys_cppclient VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Dependencies ---
# Try to find dependencies with fallback options

# Find Protobuf
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
endif()

# Find gRPC
find_package(gRPC CONFIG QUIET)
if(NOT gRPC_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPC REQUIRED grpc++)
    pkg_check_modules(GRPC_UNSECURE REQUIRED grpc++_unsecure)
    
    # Create imported targets if they don't exist
    if(NOT TARGET gRPC::grpc++)
        add_library(gRPC::grpc++ INTERFACE IMPORTED)
        target_link_libraries(gRPC::grpc++ INTERFACE ${GRPC_LIBRARIES})
        target_include_directories(gRPC::grpc++ INTERFACE ${GRPC_INCLUDE_DIRS})
        target_compile_options(gRPC::grpc++ INTERFACE ${GRPC_CFLAGS_OTHER})
    endif()
    
    if(NOT TARGET gRPC::grpc)
        add_library(gRPC::grpc INTERFACE IMPORTED)
        target_link_libraries(gRPC::grpc INTERFACE ${GRPC_UNSECURE_LIBRARIES})
        target_include_directories(gRPC::grpc INTERFACE ${GRPC_UNSECURE_INCLUDE_DIRS})
        target_compile_options(gRPC::grpc INTERFACE ${GRPC_UNSECURE_CFLAGS_OTHER})
    endif()
endif()

# Find spdlog
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(SPDLOG spdlog)
    endif()
    
    if(NOT SPDLOG_FOUND)
        # Try to find spdlog headers manually
        find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h
            PATHS /usr/include /usr/local/include /opt/homebrew/include
        )
        if(SPDLOG_INCLUDE_DIR)
            add_library(spdlog::spdlog INTERFACE IMPORTED)
            target_include_directories(spdlog::spdlog INTERFACE ${SPDLOG_INCLUDE_DIR})
        else()
            message(WARNING "spdlog not found, using header-only fallback")
            # Create a dummy target
            add_library(spdlog::spdlog INTERFACE IMPORTED)
        endif()
    endif()
endif()

# Find nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS /usr/include /usr/local/include /opt/homebrew/include
    )
    if(NLOHMANN_JSON_INCLUDE_DIR)
        add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
        target_include_directories(nlohmann_json::nlohmann_json INTERFACE ${NLOHMANN_JSON_INCLUDE_DIR})
    else()
        message(WARNING "nlohmann/json not found")
    endif()
endif()

# Find GoogleTest (optional for testing)
find_package(GTest QUIET)

# --- Protobuf Generation ---
# Assuming rl4sys.proto is in ../proto/
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
set(PROTO_FILE ${PROTO_DIR}/rl4sys.proto)
set(PROTO_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_proto)

# Ensure the output directory exists
file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Find protoc and grpc_cpp_plugin
find_program(PROTOC_EXECUTABLE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

# Generate protobuf and gRPC files manually if protobuf_generate is not available
if(EXISTS ${PROTO_FILE})
    set(PROTO_SRCS
        ${PROTO_GENERATED_DIR}/rl4sys.pb.cc
        ${PROTO_GENERATED_DIR}/rl4sys.grpc.pb.cc
    )
    set(PROTO_HDRS
        ${PROTO_GENERATED_DIR}/rl4sys.pb.h
        ${PROTO_GENERATED_DIR}/rl4sys.grpc.pb.h
    )

    add_custom_command(
        OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
        COMMAND ${PROTOC_EXECUTABLE}
        ARGS --grpc_out=${PROTO_GENERATED_DIR}
             --cpp_out=${PROTO_GENERATED_DIR}
             --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
             -I${PROTO_DIR}
             ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating protobuf and gRPC files"
    )
else()
    message(WARNING "Proto file not found at ${PROTO_FILE}, creating empty sources")
    set(PROTO_SRCS "")
    set(PROTO_HDRS "")
endif()

# --- Library Definition ---
add_library(rl4sys_client STATIC
    src/rl4sys_agent.cpp
    src/config_loader.cpp
    src/util.cpp
    ${PROTO_SRCS}
)

target_include_directories(rl4sys_client PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROTO_GENERATED_DIR}>
)

# Link libraries with proper error handling
target_link_libraries(rl4sys_client PUBLIC
    protobuf::libprotobuf
)

# Link gRPC if available
if(TARGET gRPC::grpc++)
    target_link_libraries(rl4sys_client PUBLIC gRPC::grpc++)
endif()
if(TARGET gRPC::grpc)
    target_link_libraries(rl4sys_client PUBLIC gRPC::grpc)
endif()

# Link spdlog if available
if(TARGET spdlog::spdlog)
    target_link_libraries(rl4sys_client PUBLIC spdlog::spdlog)
endif()

# Link nlohmann_json if available
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(rl4sys_client PUBLIC nlohmann_json::nlohmann_json)
endif()

# --- Testing (Optional) ---
if(GTest_FOUND)
    enable_testing()

    add_executable(cppclient_tests
        test/rl4sys_agent_test.cpp
    )

    target_link_libraries(cppclient_tests PRIVATE
        rl4sys_client
        GTest::gtest
        GTest::gtest_main
    )

    target_include_directories(cppclient_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROTO_GENERATED_DIR}
    )

    include(GoogleTest)
    gtest_discover_tests(cppclient_tests)
else()
    message(STATUS "GoogleTest not found, skipping tests")
endif()

# --- Example Executable (Optional) ---
# add_executable(test_cpp_example examples/test_cpp/test.cpp)
# target_link_libraries(test_cpp_example PRIVATE rl4sys_client)

# --- Installation (Optional) ---
# install(TARGETS rl4sys_client
#         ARCHIVE DESTINATION lib
#         LIBRARY DESTINATION lib
#         RUNTIME DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)
