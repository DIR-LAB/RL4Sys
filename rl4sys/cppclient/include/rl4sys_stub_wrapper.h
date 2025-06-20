#pragma once

#include <memory>

// Forward declarations
namespace grpc {
    class Channel;
    class ClientContext;
    class Status;
}

namespace rl4sys {
    class InitRequest;
    class InitResponse;
    class GetModelRequest;
    class ModelResponse;
    class SendTrajectoriesRequest;
    class SendTrajectoriesResponse;
}

// Interface wrapper for gRPC stub to avoid incomplete type issues in header
class RLServiceStubInterface {
public:
    explicit RLServiceStubInterface(std::shared_ptr<grpc::Channel> channel);
    ~RLServiceStubInterface();
    
    grpc::Status InitAlgorithm(grpc::ClientContext* context,
                               const rl4sys::InitRequest& request,
                               rl4sys::InitResponse* response);
    
    grpc::Status GetModel(grpc::ClientContext* context,
                          const rl4sys::GetModelRequest& request,
                          rl4sys::ModelResponse* response);
    
    grpc::Status SendTrajectories(grpc::ClientContext* context,
                                  const rl4sys::SendTrajectoriesRequest& request,
                                  rl4sys::SendTrajectoriesResponse* response);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
}; 