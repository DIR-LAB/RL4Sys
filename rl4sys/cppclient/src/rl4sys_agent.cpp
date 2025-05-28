// Placeholder for rl4sys/cppclient/src/rl4sys_agent.cpp
#include "rl4sys_agent.h"
#include "config_loader.h" // Assumes a config_loader.cpp/h exists for JSON parsing

#include <grpcpp/grpcpp.h>
// Include the generated protobuf and gRPC headers
#include "rl4sys.pb.h"
#include "rl4sys.grpc.pb.h"

#include <stdexcept> // For std::runtime_error
#include <iostream>  // For error logging (consider a proper logger)

namespace rl4sys {
namespace cppclient {

// --- RL4SysAction Implementation ---
class RL4SysAction {
public:
    // Default constructor
    RL4SysAction() : actionValue(0), actionReward(std::nullopt), done(false), version(0) {}

    // Constructor with optional parameters
    RL4SysAction(const std::vector<double>& obs, int64_t act, double reward, bool done, const std::map<std::string, std::string>& data, int ver)
        : observation(obs), actionValue(act), actionReward(reward), done(done), data(data), version(ver) {}

    void updateReward(double reward) {
        actionReward = reward;
    }

    int64_t getActionValue() const {
        return actionValue;
    }

    void setActionValue(int64_t value) {
        actionValue = value;
    }

    std::optional<double> getReward() const {
        return actionReward;
    }

    // Added: Set done flag
    void set_done(bool is_done) {
        done = is_done;
    }

    // Added: Check if reward is set
    bool is_reward_set() const {
        return actionReward.has_value();
    }

    // Added: Check if action is done
    bool is_done() const {
        return done;
    }

    // Added: Get observation
    const std::vector<double>& getObservation() const {
        return observation;
    }

    // Added: Get version
    int getVersion() const {
        return version;
    }

    // Added: Get data
    const std::map<std::string, std::string>& getData() const {
        return data;
    }

private:
    std::vector<double> observation;
    int64_t actionValue;
    std::optional<double> actionReward;
    bool done;
    std::map<std::string, std::string> data;
    int version;
};

// --- RL4SysTrajectory Implementation ---
RL4SysTrajectory::RL4SysTrajectory() {}

void RL4SysTrajectory::addAction(const RL4SysAction& action) {
    actions.push_back(action);
}

void RL4SysTrajectory::addObservation(const std::vector<double>& observation) {
    observations.push_back(observation);
}

bool RL4SysTrajectory::isEmpty() const {
    return observations.empty() && actions.empty();
}

void RL4SysTrajectory::clear() {
    observations.clear();
    actions.clear();
}

// Added: Accessors for observations and actions
const std::vector<std::vector<double>>& RL4SysTrajectory::getObservations() const {
    return observations;
}

const std::vector<RL4SysAction>& RL4SysTrajectory::getActions() const {
    return actions;
}

// --- RL4SysAgent Implementation ---

RL4SysAgent::RL4SysAgent(const std::string& configFilePath) {
    loadConfig(configFilePath);
    connect();
    if (!stub) {
        throw std::runtime_error("Failed to create gRPC stub.");
    }
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

    stub = rl4sys::RLService::NewStub(channel);

    // Optionally, check connectivity immediately
    // grpc::ClientContext context;
    // std::chrono::system_clock::time_point deadline =
    //     std::chrono::system_clock::now() + std::chrono::seconds(5); // 5-second timeout
    // context.set_deadline(deadline);
    // channel->GetState(true); // Try connecting
    // grpc_connectivity_state state = channel->GetState(false);
    // if (state != GRPC_CHANNEL_READY && state != GRPC_CHANNEL_IDLE) {
    //     std::cerr << "Warning: gRPC channel not immediately ready. State: " << state << std::endl;
    //     // Depending on policy, might want to throw or just warn
    // }

     std::cout << "gRPC channel created for address: " << config.trainServerAddress << std::endl;

}

// --- Conversion Helpers (Implementation) ---

void RL4SysAgent::convertToProtoObservation(const std::vector<double>& obs, rl4sys::Observation* protoObs) {
    if (!protoObs) return;
    protoObs->clear_features(); // Clear previous data
    for (double feature : obs) {
        protoObs->add_features(feature);
    }
    // Set other fields in protoObs if the protobuf definition has them (e.g., timestamp)
}

void RL4SysAgent::convertToProtoTrajectory(const RL4SysTrajectory& traj, rl4sys::Trajectory* protoTraj) {
    if (!protoTraj) return;
    protoTraj->clear_actions(); // Clear previous data

    // Convert actions to proto format
    for (const auto& rlAction : traj.actions) {
        rl4sys::Action* protoAction = protoTraj->add_actions();
        // Set action fields based on the actual proto definition
        // This is a simplified example - adjust based on your proto structure
        if (rlAction.is_reward_set()) {
            // Set reward if available
        }
        protoAction->set_done(rlAction.is_done());
    }

    // Set trajectory version
    protoTraj->set_version(1); // or appropriate version
}

RL4SysAction RL4SysAgent::convertFromProtoAction(const rl4sys::Action& protoAction) {
    RL4SysAction rlAction;
    // Convert from proto action to internal action
    // This needs to be implemented based on actual proto structure
    rlAction.set_done(protoAction.done());
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
    grpc::Status status = stub->SendTrajectories(&context, request, &response);

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