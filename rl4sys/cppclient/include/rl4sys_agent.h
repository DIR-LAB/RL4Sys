#pragma once

#include "rl4sys_types.h"
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

// Forward declarations for gRPC types
namespace grpc {
    class Channel;
    class ClientContext;
    class CompletionQueue;
    namespace experimental {
        enum class Compression : std::uint8_t;
    }
} // namespace grpc

// Forward declarations for protobuf messages
namespace google {
namespace protobuf {
    template<typename Key, typename T>
    class Map;
} // namespace protobuf
} // namespace google

namespace rl4sys {
    class RLService;
    class InitRequest;
    class InitResponse;
    class GetModelRequest;
    class ModelResponse;
    class SendTrajectoriesRequest;
    class SendTrajectoriesResponse;
    class Action;
    class Trajectory;
    class ParameterValue;
} // namespace rl4sys

// Opaque pointer for gRPC service stub to avoid incomplete type issues
class RLServiceStubInterface;

namespace rl4sys {
namespace cppclient {

// Forward declaration
class Logger;
class ModelManager;

/**
 * @brief The main client agent for interacting with the RL4Sys server.
 *
 * This class provides full feature parity with the Python RL4SysAgent, including:
 * - gRPC communication with InitAlgorithm, GetModel, SendTrajectories
 * - Local model management and inference
 * - Asynchronous trajectory sending with buffering
 * - Comprehensive error handling and logging
 * - Algorithm-agnostic design supporting PPO, DQN, etc.
 */
class RL4SysAgent {
public:
    // Use a custom deleter class to avoid incomplete type issues
    struct StubDeleter {
        void operator()(void* ptr);
    };

    /**
     * @brief Constructs an RL4SysAgent with configuration file.
     * @param configFilePath Path to the JSON configuration file.
     * @param debug Enable debug logging (optional, default: false).
     * @throws std::runtime_error if configuration loading or server connection fails.
     */
    explicit RL4SysAgent(const std::string& configFilePath, bool debug = false);

    /**
     * @brief Destructor. Cleanly shuts down threads and gRPC connection.
     */
    ~RL4SysAgent();

    // Disable copy and move semantics for thread safety
    RL4SysAgent(const RL4SysAgent&) = delete;
    RL4SysAgent& operator=(const RL4SysAgent&) = delete;
    RL4SysAgent(RL4SysAgent&&) = delete;
    RL4SysAgent& operator=(RL4SysAgent&&) = delete;

    /**
     * @brief Request an action for the given observation.
     * 
     * This is the main interface for generating actions. It handles:
     * - Creating new trajectories or continuing existing ones
     * - Running local model inference to generate actions
     * - Managing trajectory lifecycle
     * 
     * @param trajectory Current trajectory (nullptr to start new one)
     * @param observation Observation vector (must match model input size)
     * @param mask Optional mask vector (can be same length as observation or empty)
     * @return Pair of (updated_trajectory, action) or nullopt on failure
     */
    std::optional<std::pair<std::shared_ptr<RL4SysTrajectory>, RL4SysAction>> 
    requestForAction(std::shared_ptr<RL4SysTrajectory> trajectory,
                     const std::vector<float>& observation,
                     const std::vector<float>& mask = std::vector<float>());

    /**
     * @brief Add an action to the trajectory.
     * 
     * @param trajectory The trajectory to update
     * @param action The action to add
     */
    void addToTrajectory(std::shared_ptr<RL4SysTrajectory> trajectory, 
                         const RL4SysAction& action);

    /**
     * @brief Update the reward for an action.
     * 
     * @param action The action to update
     * @param reward The reward value
     */
    void updateActionReward(RL4SysAction& action, double reward);

    /**
     * @brief Mark the end of a trajectory and queue it for sending.
     * 
     * This will add the trajectory to the sending buffer. The trajectory
     * will be sent asynchronously when enough trajectories are accumulated.
     * 
     * @param trajectory The completed trajectory
     * @param lastAction The final action in the trajectory
     * @return True if trajectory was successfully queued for sending
     */
    bool markEndOfTrajectory(std::shared_ptr<RL4SysTrajectory> trajectory, 
                             RL4SysAction& lastAction);

    /**
     * @brief Force sending all buffered trajectories immediately.
     * 
     * @return True if all trajectories were sent successfully
     */
    bool flushTrajectories();

    /**
     * @brief Close the agent and cleanup resources.
     * 
     * This will stop all background threads and close the gRPC connection.
     */
    void close();

    /**
     * @brief Get the current model version.
     */
    int32_t getCurrentModelVersion() const;

    /**
     * @brief Get agent configuration.
     */
    const AgentConfig& getConfig() const { return config_; }

private:
    /**
     * @brief Initialize the server-side algorithm.
     * 
     * Calls InitAlgorithm gRPC service to set up the algorithm on the server.
     */
    void initializeServerAlgorithm();

    /**
     * @brief Get the latest model from the server.
     * 
     * Calls GetModel gRPC service and updates local model cache.
     * 
     * @param expectedVersion Expected model version (-1 for initial fetch)
     * @return True if model was updated successfully
     */
    bool getModelFromServer(int32_t expectedVersion = -1);

    /**
     * @brief Send trajectories to the server.
     * 
     * Calls SendTrajectories gRPC service with compression.
     * 
     * @param trajectories Vector of trajectories to send
     * @return True if trajectories were sent successfully
     */
    bool sendTrajectoriesToServer(const std::vector<std::shared_ptr<RL4SysTrajectory>>& trajectories);

    /**
     * @brief Convert RL4SysAction to protobuf Action.
     */
    void convertToProtoAction(const RL4SysAction& action, rl4sys::Action* protoAction);

    /**
     * @brief Convert RL4SysTrajectory to protobuf Trajectory.
     */
    void convertToProtoTrajectory(const RL4SysTrajectory& trajectory, rl4sys::Trajectory* protoTrajectory);

    /**
     * @brief Convert algorithm parameters to protobuf format.
     */
    void convertAlgorithmParameters(const std::map<std::string, std::variant<int32_t, double, std::string, bool>>& params,
                                    google::protobuf::Map<std::string, rl4sys::ParameterValue>& protoParams);

    /**
     * @brief Background thread worker for sending trajectories.
     */
    void sendThreadWorker();

    /**
     * @brief Check if we should send trajectories based on buffer size.
     */
    bool shouldSendTrajectories();

    /**
     * @brief Create gRPC channel with compression and timeout settings.
     */
    void createGrpcChannel();

    // Configuration and core state
    AgentConfig config_;
    std::unique_ptr<Logger> logger_;
    
    // gRPC communication
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<RLServiceStubInterface> stub_;
    
    // Model management
    std::unique_ptr<ModelManager> model_manager_;
    std::atomic<int32_t> local_model_version_;
    
    // Trajectory management is handled via the send_queue_
    
    // Threading for async trajectory sending
    std::unique_ptr<std::thread> send_thread_;
    std::atomic<bool> stop_flag_;
    std::mutex send_mutex_;
    std::condition_variable send_cv_;
    std::queue<std::shared_ptr<RL4SysTrajectory>> send_queue_;
    
    // Thread safety
    mutable std::mutex state_mutex_;
    
    // Connection state
    std::atomic<bool> initialized_;
    std::atomic<bool> connected_;
};

} // namespace cppclient
} // namespace rl4sys