#!/bin/bash

# Generate protobuf files with Python package structure
python -m grpc_tools.protoc \
    -I . \
    --python_out=. \
    --grpc_python_out=. \
    --proto_path=. \
    rl4sys.proto

echo "Protobuf files generated successfully!" 