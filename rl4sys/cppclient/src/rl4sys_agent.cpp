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

    stub = rl4sys_proto::RL4SysService::NewStub(channel);

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

void RL4SysAgent::convertToProtoObservation(const std::vector<double>& obs, rl4sys_proto::Observation* protoObs) {
    if (!protoObs) return;
    protoObs->clear_features(); // Clear previous data
    for (double feature : obs) {
        protoObs->add_features(feature);
    }
    // Set other fields in protoObs if the protobuf definition has them (e.g., timestamp)
}

void RL4SysAgent::convertToProtoTrajectory(const RL4SysTrajectory& traj, rl4sys_proto::Trajectory* protoTraj) {
    if (!protoTraj) return;
    protoTraj->clear_observations();
    protoTraj->clear_actions();
    protoTraj->clear_rewards(); // Assuming rewards are stored separately in proto

    // This assumes a specific structure for the Trajectory proto message.
    // Adjust according to your actual rl4sys.proto definition.

    // Example: Assuming Observations are stored sequentially
    for (const auto& obsVec : traj.observations) {
        rl4sys_proto::Observation* pObs = protoTraj->add_observations();
        convertToProtoObservation(obsVec, pObs);
    }

    // Example: Assuming Actions and Rewards are stored sequentially
    for (const auto& rlAction : traj.actions) {
         protoTraj->add_actions(rlAction.getActionValue()); // Add action value
         if (rlAction.actionReward.has_value()) {
            protoTraj->add_rewards(rlAction.actionReward.value());
         } else {
             // Handle missing reward? Maybe add a default value (e.g., 0.0) or indicator
             protoTraj->add_rewards(0.0); // Placeholder
         }
    }

    // Set other trajectory metadata if defined in the proto (e.g., client_id, episode_id)
    protoTraj->set_client_id(config.clientId);

}

RL4SysAction RL4SysAgent::convertFromProtoAction(const rl4sys_proto::Action& protoAction) {
    RL4SysAction rlAction;
    // Assuming protoAction has a field like 'action_value'
    rlAction.actionValue = protoAction.action_value();
    // If protoAction contains reward or other fields, populate rlAction accordingly
    return rlAction;
}


// --- Core Agent Logic ---

std::optional<std::pair<RL4SysTrajectory, RL4SysAction>> RL4SysAgent::requestForAction(
    std::optional<RL4SysTrajectory>& currentTrajectoryOpt,
    const std::vector<double>& observation)
{
    rl4sys_proto::ActionRequest request;
    rl4sys_proto::ActionResponse response;
    grpc::ClientContext context;

    // Set client ID in request
    request.set_client_id(config.clientId);

    // Add the latest observation
    convertToProtoObservation(observation, request.mutable_last_observation());

    // Include the current partial trajectory if it exists
    // This depends heavily on how the server expects partial trajectories.
    // Option 1: Server expects the *entire* history so far in each request.
    // Option 2: Server only needs the *latest* state/observation.
    // Option 3: Server expects incremental updates (more complex).
    // Assuming Option 1 for this example:
    if (currentTrajectoryOpt.has_value()) {
         convertToProtoTrajectory(currentTrajectoryOpt.value(), request.mutable_current_trajectory());
    }

    // Set a deadline for the RPC
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::seconds(10); // 10-second timeout
    context.set_deadline(deadline);

    // Make the gRPC call
     std::cout << "Sending RequestForAction to server..." << std::endl;
    grpc::Status status = stub->RequestForAction(&context, request, &response);

    if (status.ok()) {
         std::cout << "Received action from server." << std::endl;
        RL4SysAction newAction = convertFromProtoAction(response.action());

        // Initialize or update the trajectory
        if (!currentTrajectoryOpt.has_value()) {
            currentTrajectoryOpt = RL4SysTrajectory();
        }
        // The trajectory is typically updated *after* the action is taken and reward received.
        // So, we add the *observation* now, but the action/reward later.
        currentTrajectoryOpt.value().addObservation(observation);


        return std::make_pair(currentTrajectoryOpt.value(), newAction); // Return by value might be inefficient for large trajectories

    } else {
         std::cerr << "gRPC RequestForAction failed: (" << status.error_code() << ") "
                  << status.error_message() << std::endl;
        // Handle specific errors (e.g., UNAVAILABLE, DEADLINE_EXCEEDED) if needed
        return std::nullopt;
    }
}


void RL4SysAgent::addToTrajectory(RL4SysTrajectory& trajectory, const RL4SysAction& action) {
    // This function assumes the action object has already been updated with its reward
    // by the calling code (e.g., lunar_lander simulation loop) via action.updateReward().
    trajectory.addAction(action);
    // Potentially check if trajectory length triggers sending based on config.sendFrequency
}


bool RL4SysAgent::markEndOfTrajectory(RL4SysTrajectory& trajectory, RL4SysAction& lastAction) {
     // Ensure the last action (and its potential reward) is added before sending
     trajectory.addAction(lastAction); // Add the final action

     rl4sys_proto::SendTrajectoryRequest request;
     rl4sys_proto::SendTrajectoryResponse response;
     grpc::ClientContext context;

     // Convert the completed trajectory to the protobuf format
     convertToProtoTrajectory(trajectory, request.mutable_trajectory());
     request.mutable_trajectory()->set_done(true); // Mark trajectory as complete


     // Set a deadline
     std::chrono::system_clock::time_point deadline =
         std::chrono::system_clock::now() + std::chrono::seconds(15); // Longer timeout for potentially larger data
     context.set_deadline(deadline);

     // Send the trajectory
      std::cout << "Sending completed trajectory to server..." << std::endl;
     grpc::Status status = stub->SendTrajectory(&context, request, &response);

     if (status.ok()) {
          std::cout << "Trajectory sent successfully." << std::endl;
         // Clear the trajectory state after successful sending
         trajectory.clear();
         return true;
     } else {
          std::cerr << "gRPC SendTrajectory failed: (" << status.error_code() << ") "
                   << status.error_message() << std::endl;
         // Decide how to handle failure - retry? Log and discard?
         // For now, just return false. The trajectory data remains in the passed object.
         return false;
     }
}


} // namespace cppclient
} // namespace rl4sys