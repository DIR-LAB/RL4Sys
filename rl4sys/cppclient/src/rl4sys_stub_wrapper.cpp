#include "rl4sys_stub_wrapper.h"
#include "rl4sys.grpc.pb.h"
#include <grpcpp/grpcpp.h>

// Implementation class using pImpl idiom
class RLServiceStubInterface::Impl {
public:
    explicit Impl(std::shared_ptr<grpc::Channel> channel) 
        : stub_(rl4sys::RLService::NewStub(channel)) {}
    
    std::unique_ptr<rl4sys::RLService::Stub> stub_;
};

// Interface implementation
RLServiceStubInterface::RLServiceStubInterface(std::shared_ptr<grpc::Channel> channel)
    : pImpl(std::make_unique<Impl>(channel)) {}

RLServiceStubInterface::~RLServiceStubInterface() = default;

grpc::Status RLServiceStubInterface::InitAlgorithm(grpc::ClientContext* context,
                                                  const rl4sys::InitRequest& request,
                                                  rl4sys::InitResponse* response) {
    return pImpl->stub_->InitAlgorithm(context, request, response);
}

grpc::Status RLServiceStubInterface::GetModel(grpc::ClientContext* context,
                                             const rl4sys::GetModelRequest& request,
                                             rl4sys::ModelResponse* response) {
    return pImpl->stub_->GetModel(context, request, response);
}

grpc::Status RLServiceStubInterface::SendTrajectories(grpc::ClientContext* context,
                                                     const rl4sys::SendTrajectoriesRequest& request,
                                                     rl4sys::SendTrajectoriesResponse* response) {
    return pImpl->stub_->SendTrajectories(context, request, response);
} 