#pragma once

#include <vector>
#include <string>
#include <optional>
#include <map>
#include <memory>
#include <mutex>
#include <cstdint>
#include <variant>

namespace rl4sys {
namespace cppclient {

/**
 * @brief Represents a single action in the RL4Sys framework.
 * 
 * This matches the proto Action message structure with:
 * - obs: serialized observation tensor
 * - action: serialized action tensor  
 * - reward: serialized reward tensor
 * - done: episode termination flag
 * - mask: serialized mask tensor (optional)
 * - extra_data: additional algorithm-specific data
 */
class RL4SysAction {
public:
    RL4SysAction();
    
    /**
     * @brief Constructor with observation and action data.
     * @param obs_data Serialized observation tensor bytes
     * @param action_data Serialized action tensor bytes
     * @param reward_value Reward value (default: 0.0)
     * @param is_done Episode termination flag (default: false)
     * @param mask_data Serialized mask tensor bytes (optional, default: empty)
     * @param version Model version used for this action (default: 0)
     */
    RL4SysAction(const std::vector<uint8_t>& obs_data,
                 const std::vector<uint8_t>& action_data,
                 double reward_value = 0.0,
                 bool is_done = false,
                 const std::vector<uint8_t>& mask_data = std::vector<uint8_t>(),
                 int32_t version = 0);
    
    // Getters
    const std::vector<uint8_t>& getObservation() const { return obs_bytes_; }
    const std::vector<uint8_t>& getAction() const { return action_bytes_; }
    double getReward() const { return reward_; }
    bool isDone() const { return done_; }
    const std::vector<uint8_t>& getMask() const { return mask_bytes_; }
    int32_t getVersion() const { return version_; }
    const std::map<std::string, std::vector<uint8_t>>& getExtraData() const { return extra_data_; }
    
    // Setters
    void setObservation(const std::vector<uint8_t>& obs_data) { obs_bytes_ = obs_data; }
    void setAction(const std::vector<uint8_t>& action_data) { action_bytes_ = action_data; }
    void setReward(double reward) { reward_ = reward; }
    void setDone(bool done) { done_ = done; }
    void setMask(const std::vector<uint8_t>& mask_data) { mask_bytes_ = mask_data; }
    void setVersion(int32_t version) { version_ = version; }
    
    /**
     * @brief Add extra data (e.g., logp_a, v for PPO)
     * @param key Data key name
     * @param data Serialized data bytes
     */
    void addExtraData(const std::string& key, const std::vector<uint8_t>& data);
    
    /**
     * @brief Clear all data to free memory
     */
    void clear();

private:
    std::vector<uint8_t> obs_bytes_;
    std::vector<uint8_t> action_bytes_;
    double reward_;
    bool done_;
    std::vector<uint8_t> mask_bytes_;
    int32_t version_;
    std::map<std::string, std::vector<uint8_t>> extra_data_;
};

/**
 * @brief Represents a trajectory of actions in the RL4Sys framework.
 * 
 * This matches the proto Trajectory message structure with:
 * - actions: repeated Action messages
 * - version: model version for this trajectory
 */
class RL4SysTrajectory {
public:
    RL4SysTrajectory();
    explicit RL4SysTrajectory(int32_t version);
    
    /**
     * @brief Add an action to the trajectory
     * @param action The RL4SysAction to add
     */
    void addAction(const RL4SysAction& action);
    void addAction(RL4SysAction&& action);
    
    /**
     * @brief Mark trajectory as completed
     */
    void markCompleted() { completed_ = true; }
    
    /**
     * @brief Check if trajectory is completed
     */
    bool isCompleted() const { return completed_; }
    
    /**
     * @brief Check if trajectory is valid (has at least one action)
     */
    bool isValid() const { return !actions_.empty(); }
    
    /**
     * @brief Check if trajectory is empty
     */
    bool isEmpty() const { return actions_.empty(); }
    
    /**
     * @brief Get number of actions in trajectory
     */
    size_t size() const { return actions_.size(); }
    
    /**
     * @brief Clear all actions to free memory
     */
    void clear();
    
    // Getters
    const std::vector<RL4SysAction>& getActions() const { return actions_; }
    std::vector<RL4SysAction>& getActions() { return actions_; }
    int32_t getVersion() const { return version_; }
    
    // Setters  
    void setVersion(int32_t version) { version_ = version; }

private:
    std::vector<RL4SysAction> actions_;
    int32_t version_;
    bool completed_;
};

/**
 * @brief Configuration for the RL4Sys agent.
 * 
 * This matches the JSON configuration structure used by the Python client.
 */
struct AgentConfig {
    // Core identification
    std::string client_id = "default_client";
    std::string algorithm_name = "PPO";
    std::string algorithm_type = "onpolicy";
    
    // Server connection
    std::string train_server_address = "localhost:50051";
    
    // Algorithm parameters (matches luna_conf.json structure)
    std::map<std::string, std::variant<int32_t, double, std::string, bool>> algorithm_parameters;
    
    // Client behavior settings
    double act_limit = 1.0;
    int32_t max_traj_length = 1000;
    int32_t send_frequency = 10; // How many completed trajectories before sending
    
    // Connection settings
    int32_t connection_timeout_seconds = 10;
    int32_t request_timeout_seconds = 30;
    bool enable_compression = true;
    
    // Debug settings
    bool debug = false;
    std::string log_level = "INFO";
};

/**
 * @brief Utility functions for serialization/deserialization
 */
namespace utils {
    /**
     * @brief Serialize a float value to bytes
     * @param value The float value to serialize
     * @return Vector of bytes representing the serialized value
     */
    std::vector<uint8_t> serializeFloat(float value);
    
    /**
     * @brief Serialize a vector of floats to bytes
     * @param values The vector of floats to serialize
     * @return Vector of bytes representing the serialized values
     */
    std::vector<uint8_t> serializeFloatVector(const std::vector<float>& values);
    
    /**
     * @brief Deserialize bytes to a float value
     * @param data The bytes to deserialize
     * @return The deserialized float value
     */
    float deserializeFloat(const std::vector<uint8_t>& data);
    
    /**
     * @brief Deserialize bytes to a vector of floats
     * @param data The bytes to deserialize
     * @return The deserialized vector of floats
     */
    std::vector<float> deserializeFloatVector(const std::vector<uint8_t>& data);
    
    /**
     * @brief Serialize an int64 action to bytes
     * @param action The action value to serialize
     * @return Vector of bytes representing the serialized action
     */
    std::vector<uint8_t> serializeAction(int64_t action);
    
    /**
     * @brief Deserialize bytes to an int64 action
     * @param data The bytes to deserialize
     * @return The deserialized action value
     */
    int64_t deserializeAction(const std::vector<uint8_t>& data);
}

} // namespace cppclient
} // namespace rl4sys