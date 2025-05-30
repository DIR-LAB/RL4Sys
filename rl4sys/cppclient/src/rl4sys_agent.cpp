// Placeholder for rl4sys/cppclient/src/rl4sys_agent.cpp
#include "rl4sys_agent.h"
#include "config_loader.h" // Assumes a config_loader.cpp/h exists for JSON parsing
#include "util.h" // For serialization functions

#include <grpcpp/grpcpp.h>
// Include the generated protobuf and gRPC headers
#include "rl4sys.pb.h"
#include "rl4sys.grpc.pb.h"

#include <stdexcept> // For std::runtime_error
#include <iostream>  // For error logging (consider a proper logger)

namespace rl4sys {
namespace cppclient {

// Define the StubDeleter implementation
void RL4SysAgent::StubDeleter::operator()(void* ptr) {
    delete static_cast<rl4sys::RLService::Stub*>(ptr);
}

// Helper function to get the gRPC stub
rl4sys::RLService::Stub* getStub(const std::unique_ptr<void, RL4SysAgent::StubDeleter>& stub_ptr) {
    return static_cast<rl4sys::RLService::Stub*>(stub_ptr.get());
}

// --- RL4SysAgent Implementation ---

RL4SysAgent::RL4SysAgent(const std::string& configFilePath) {
    loadConfig(configFilePath);
    connect();
    if (!stub) {
        throw std::runtime_error("Failed to create gRPC stub.");
    }
    
    // Initialize the algorithm on the server
    initializeAlgorithm();
    
    std::cout << "RL4SysAgent initialized. Connected to " << config.trainServerAddress << std::endl;
}

RL4SysAgent::~RL4SysAgent() {
    // gRPC channel and stub are managed by smart pointers (shared_ptr/unique_ptr)
    // Automatic cleanup happens here.
     std::cout << "RL4SysAgent destroyed." << std::endl;
}

void RL4SysAgent::loadConfig(const std::string& configFilePath) {
    // Use the ConfigLoader utility
    try {
        config = ConfigLoader::loadFromFile(configFilePath);
         std::cout << "Configuration loaded successfully from: " << configFilePath << std::endl;
    } catch (const std::exception& e) {
         std::cerr << "Failed to load configuration: " << e.what() << std::endl;
        throw; // Re-throw the exception
    }
}

void RL4SysAgent::connect() {
    grpc::ChannelArguments args;
    // Set channel arguments if needed (e.g., keepalive, message size limits)
    // args.SetMaxSendMessageSize(-1); // Example: Unlimited message size
    // args.SetMaxReceiveMessageSize(-1);

    channel = grpc::CreateCustomChannel(
        config.trainServerAddress,
        grpc::InsecureChannelCredentials(), // Use secure credentials in production
        args
    );

    if (!channel) {
        throw std::runtime_error("Failed to create gRPC channel to " + config.trainServerAddress);
    }

    // Create the stub and store it as void* with custom deleter
    auto grpc_stub = rl4sys::RLService::NewStub(channel);
    stub = std::unique_ptr<void, RL4SysAgent::StubDeleter>(grpc_stub.release(), StubDeleter());

    std::cout << "gRPC channel created for address: " << config.trainServerAddress << std::endl;
}

void RL4SysAgent::initializeAlgorithm() {
    rl4sys::InitRequest request;
    rl4sys::InitResponse response;
    grpc::ClientContext context;

    // Set client ID and algorithm name from config
    request.set_client_id(config.clientId);
    request.set_algorithm_name("PPO"); // Default to PPO, could be made configurable

    // Add algorithm parameters from config
    // For now, add some basic parameters - this could be expanded based on config
    auto* params = request.mutable_algorithm_parameters();
    
    // Add batch_size parameter
    rl4sys::ParameterValue batch_size;
    batch_size.set_int_value(512);
    (*params)["batch_size"] = batch_size;
    
    // Add act_dim parameter
    rl4sys::ParameterValue act_dim;
    act_dim.set_int_value(4);
    (*params)["act_dim"] = act_dim;
    
    // Add input_size parameter
    rl4sys::ParameterValue input_size;
    input_size.set_int_value(8);
    (*params)["input_size"] = input_size;
    
    // Add seed parameter
    rl4sys::ParameterValue seed;
    seed.set_int_value(0);
    (*params)["seed"] = seed;

    // Set a deadline
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::seconds(10);
    context.set_deadline(deadline);

    // Call InitAlgorithm
    std::cout << "Initializing algorithm on server..." << std::endl;
    grpc::Status status = getStub(stub)->InitAlgorithm(&context, request, &response);

    if (status.ok()) {
        if (response.is_success()) {
            std::cout << "Algorithm initialized successfully: " << response.message() << std::endl;
        } else {
            std::cerr << "Algorithm initialization failed: " << response.message() << std::endl;
            throw std::runtime_error("Algorithm initialization failed: " + response.message());
        }
    } else {
        std::cerr << "gRPC InitAlgorithm failed: (" << status.error_code() << ") "
                  << status.error_message() << std::endl;
        throw std::runtime_error("Failed to initialize algorithm: " + status.error_message());
    }
}

// --- Conversion Helpers (Implementation) ---

void RL4SysAgent::convertToProtoObservation(const std::vector<double>& obs, rl4sys::Observation* protoObs) {
    // Note: Based on the proto file, there's no Observation message with features
    // This method may need to be updated based on actual proto structure
    if (!protoObs) return;
    // Implementation depends on actual proto structure
}

void RL4SysAgent::convertToProtoTrajectory(const RL4SysTrajectory& traj, rl4sys::Trajectory* protoTraj) {
    if (!protoTraj) return;
    protoTraj->clear_actions(); // Clear previous data

    // Convert actions to proto format using the public accessor
    for (const auto& rlAction : traj.getActions()) {
        rl4sys::Action* protoAction = protoTraj->add_actions();
        
        // Serialize observation as bytes
        if (!rlAction.getObservation().empty()) {
            std::vector<float> obs_float(rlAction.getObservation().begin(), rlAction.getObservation().end());
            auto obs_bytes = serialize_tensor(obs_float);
            protoAction->set_obs(obs_bytes.data(), obs_bytes.size());
        }
        
        // Serialize action value as bytes
        std::vector<float> action_vec = {static_cast<float>(rlAction.getActionValue())};
        auto action_bytes = serialize_tensor(action_vec);
        protoAction->set_action(action_bytes.data(), action_bytes.size());
        
        // Serialize reward as bytes if available
        if (rlAction.is_reward_set()) {
            std::vector<float> reward_vec = {static_cast<float>(rlAction.getReward().value())};
            auto reward_bytes = serialize_tensor(reward_vec);
            protoAction->set_reward(reward_bytes.data(), reward_bytes.size());
        }
        
        // Set done flag
        protoAction->set_done(rlAction.is_done());
        
        // Set extra data if any
        auto data = rlAction.getData();
        for (const auto& [key, value] : data) {
            std::vector<float> value_vec = {static_cast<float>(std::stof(value))};
            auto value_bytes = serialize_tensor(value_vec);
            (*protoAction->mutable_extra_data())[key] = std::string(value_bytes.begin(), value_bytes.end());
        }
    }

    // Set trajectory version
    protoTraj->set_version(1); // or appropriate version
}

RL4SysAction RL4SysAgent::convertFromProtoAction(const rl4sys::Action& protoAction) {
    RL4SysAction rlAction;
    // Convert from proto action to internal action
    // This needs to be implemented based on actual proto structure
    // The proto Action uses bytes for serialized tensors, not simple fields
    return rlAction;
}

// --- Core Agent Logic ---

std::optional<std::pair<RL4SysTrajectory, RL4SysAction>> RL4SysAgent::requestForAction(
    std::optional<RL4SysTrajectory>& currentTrajectoryOpt,
    const std::vector<double>& observation)
{
    // For now, return a simple mock response since the actual proto doesn't have RequestForAction
    // This needs to be implemented based on the actual server API
    RL4SysAction mockAction;
    mockAction.setActionValue(0); // Default action
    
    if (!currentTrajectoryOpt.has_value()) {
        currentTrajectoryOpt = RL4SysTrajectory();
    }
    currentTrajectoryOpt.value().addObservation(observation);
    
    return std::make_pair(currentTrajectoryOpt.value(), mockAction);
}

void RL4SysAgent::addToTrajectory(RL4SysTrajectory& trajectory, const RL4SysAction& action) {
    trajectory.addAction(action);
}

bool RL4SysAgent::markEndOfTrajectory(RL4SysTrajectory& trajectory, RL4SysAction& lastAction) {
    trajectory.addAction(lastAction);

    rl4sys::SendTrajectoriesRequest request;
    rl4sys::SendTrajectoriesResponse response;
    grpc::ClientContext context;

    // Set client ID
    request.set_client_id(config.clientId);
    
    // Add the trajectory
    rl4sys::Trajectory* protoTraj = request.add_trajectories();
    convertToProtoTrajectory(trajectory, protoTraj);

    // Set a deadline
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::seconds(15);
    context.set_deadline(deadline);

    // Send the trajectory using the actual RPC method from proto
    std::cout << "Sending completed trajectory to server..." << std::endl;
    grpc::Status status = getStub(stub)->SendTrajectories(&context, request, &response);

    if (status.ok()) {
        std::cout << "Trajectory sent successfully." << std::endl;
        trajectory.clear();
        return true;
    } else {
        std::cerr << "gRPC SendTrajectories failed: (" << status.error_code() << ") "
                  << status.error_message() << std::endl;
        return false;
    }
}

} // namespace cppclient
} // namespace rl4sys