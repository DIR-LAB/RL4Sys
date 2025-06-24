#include <grpcpp/grpcpp.h>
#include "helloworld.grpc.pb.h"
#include "helloworld.pb.h"
#include <iostream>

using helloworld::ProcessingServices;
using helloworld::Point3;
using helloworld::Numeric;

class ProcessingImpl final : public ProcessingServices::Service {
public:
    // 方法签名必须与生成代码完全一致（包括大小写）
    ::grpc::Status computeSum(::grpc::ServerContext* /*context*/,
                              const Point3* request,
                              Numeric* reply) override
    {
        reply->set_value(request->x() + request->y() + request->z());
        return ::grpc::Status::OK;
    }
};  // ← 别忘了分号

int main() {
    ProcessingImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort("0.0.0.0:9999",
                             grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on 0.0.0.0:9999\n";
    server->Wait();
    return 0;
}
