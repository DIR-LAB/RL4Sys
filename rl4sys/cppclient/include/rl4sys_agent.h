// Placeholder for rl4sys/cppclient/include/rl4sys_agent.h
#pragma once

#include "rl4sys_types.h"
#include <string>
#include <vector>
#include <memory> // For std::unique_ptr
#include <optional>

// Forward declarations for gRPC types
namespace grpc {
    class Channel;
    class ClientContext;
} // namespace grpc

// Forward declarations for Protobuf message and service types
namespace rl4sys {
    class RLService; // The gRPC service (matches proto file)
    class Observation;
    class ActionRequest;
    class ActionResponse;
    class Trajectory;
    class SendTrajectoryRequest;
    class SendTrajectoryResponse;
    class Action;
    class GetModelRequest;
    class ModelResponse;
    class SendTrajectoriesRequest;
    class SendTrajectoriesResponse;
} // namespace rl4sys


namespace rl4sys {
namespace cppclient {

/**
 * @brief Configuration settings for the RL4SysAgent.
 */
struct AgentConfig {
    std::string clientId = "default_client";
    std::string trainServerAddress = "localhost:50051";
    int sendFrequency = 10; // Example: How often to send trajectories
    double actLimit = 1.0;
    // Add other relevant configuration parameters based on luna_conf.json
    // e.g., algorithm details if needed by the client directly
};

/**
 * @brief The main client agent for interacting with the RL4Sys server.
 *
 * This class handles communication with the gRPC server to request actions
 * based on observations and send completed trajectories for training.
 * It follows the RAII principle for managing the gRPC connection.
 */
class RL4SysAgent {
public:
    // Use a custom deleter class to avoid incomplete type issues
    struct StubDeleter {
        void operator()(void* ptr);
    };

    /**
     * @brief Constructs an RL4SysAgent.
     * @param configFilePath Path to the JSON configuration file.
     * @throws std::runtime_error if configuration loading or gRPC connection fails.
     */
    explicit RL4SysAgent(const std::string& configFilePath);

    /**
     * @brief Destructor. Cleans up gRPC resources.
     */
    ~RL4SysAgent();

    // Disable copy and move semantics for simplicity with gRPC resources
    RL4SysAgent(const RL4SysAgent&) = delete;
    RL4SysAgent& operator=(const RL4SysAgent&) = delete;
    RL4SysAgent(RL4SysAgent&&) = delete;
    RL4SysAgent& operator=(RL4SysAgent&&) = delete;


    /**
     * @brief Requests an action from the RL4Sys server based on the current observation.
     *
     * Sends the current partial trajectory (if any) and the latest observation
     * to the server and receives the next action to take.
     *
     * @param currentTrajectory The current trajectory being built. Can be nullopt initially.
     * @param observation The latest observation from the environment. Example type, adjust as needed.
     * @return A pair containing the updated trajectory and the suggested action.
     *         Returns nullopt for trajectory/action if the request fails.
     * @throws std::runtime_error on gRPC communication errors.
     */
     std::optional<std::pair<RL4SysTrajectory, RL4SysAction>> requestForAction(
        std::optional<RL4SysTrajectory>& currentTrajectory,
        const std::vector<double>& observation); // Use appropriate observation type


    /**
     * @brief Adds a completed action (with reward) to the trajectory.
     *
     * This method is typically called after the environment step, once the
     * reward for the `lastAction` is known.
     *
     * @param trajectory The trajectory to add the action to.
     * @param action The action object (potentially updated with reward) to add.
     */
    void addToTrajectory(RL4SysTrajectory& trajectory, const RL4SysAction& action);


     /**
      * @brief Marks the end of the current trajectory and sends it to the server.
      *
      * This should be called when an episode terminates or reaches a maximum step limit.
      * The internal state related to the current trajectory might be reset after sending.
      *
      * @param trajectory The completed trajectory to send.
      * @param lastAction The final action taken in the trajectory (may need reward update).
      * @return True if the trajectory was sent successfully, false otherwise.
      */
     bool markEndOfTrajectory(RL4SysTrajectory& trajectory, RL4SysAction& lastAction);


private:
    /**
     * @brief Loads configuration from the specified JSON file.
     * @param configFilePath Path to the JSON configuration file.
     */
    void loadConfig(const std::string& configFilePath);

    /**
     * @brief Establishes the gRPC connection to the server.
     */
    void connect();

    /**
     * @brief Initializes the algorithm on the server.
     */
    void initializeAlgorithm();

    /**
     * @brief Helper to convert internal observation type to protobuf Observation.
     * @param obs Internal observation data.
     * @param protoObs Pointer to the protobuf message to fill.
     */
    void convertToProtoObservation(const std::vector<double>& obs, rl4sys::Observation* protoObs);

     /**
      * @brief Helper to convert internal trajectory type to protobuf Trajectory.
      * @param traj Internal trajectory data.
      * @param protoTraj Pointer to the protobuf message to fill.
      */
     void convertToProtoTrajectory(const RL4SysTrajectory& traj, rl4sys::Trajectory* protoTraj);

     /**
      * @brief Helper to convert protobuf Action to internal RL4SysAction type.
      * @param protoAction The protobuf Action message.
      * @return The internal RL4SysAction object.
      */
     RL4SysAction convertFromProtoAction(const rl4sys::Action& protoAction);


    AgentConfig config;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<void, StubDeleter> stub;
    
    // State related to the current trajectory being built, if needed client-side
    // std::optional<RL4SysTrajectory> ongoingTrajectory;
};

} // namespace cppclient
} // namespace rl4sys