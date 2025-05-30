#include <grpcpp/grpcpp.h>
#include "build/helloworld.grpc.pb.h"
#include "build/helloworld.pb.h"
#include <iostream>

int main() {
    ProcessingService::Stub stub(grpc::CreateChannel("localhost:9999", grpc::InsecureChannelCredentials()));
    Point3 request;
    request.set_x(1.0);
    request.set_y(2.0);
    request.set_z(3.0);
    Numeric response;
    stub.computeSum(&request, &response);
    std::cout << "Sum: " << response.value() << std::endl;
    return 0;
}