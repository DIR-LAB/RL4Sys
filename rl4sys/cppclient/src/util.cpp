#include "util.h"
#include <vector>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace rl4sys {
namespace cppclient {

// Helper function to convert float to bytes
std::vector<uint8_t> float_to_bytes(float value) {
    std::vector<uint8_t> bytes(sizeof(float));
    std::memcpy(bytes.data(), &value, sizeof(float));
    return bytes;
}

// Helper function to convert bytes to float
float bytes_to_float(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != sizeof(float)) {
        throw std::runtime_error("Invalid byte size for float conversion");
    }
    float value;
    std::memcpy(&value, bytes.data(), sizeof(float));
    return value;
}

// Serialize tensor (vector of floats) to bytes
std::vector<uint8_t> serialize_tensor(const std::vector<float>& tensor) {
    if (tensor.empty()) {
        return std::vector<uint8_t>();
    }
    
    std::vector<uint8_t> result;
    result.reserve(tensor.size() * sizeof(float));
    
    for (float value : tensor) {
        auto bytes = float_to_bytes(value);
        result.insert(result.end(), bytes.begin(), bytes.end());
    }
    
    return result;
}

// Deserialize bytes to tensor (vector of floats)
std::vector<float> deserialize_tensor(const std::vector<uint8_t>& tensor_bytes) {
    if (tensor_bytes.empty()) {
        return std::vector<float>();
    }
    
    if (tensor_bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("Invalid tensor byte size");
    }
    
    std::vector<float> result;
    result.reserve(tensor_bytes.size() / sizeof(float));
    
    for (size_t i = 0; i < tensor_bytes.size(); i += sizeof(float)) {
        std::vector<uint8_t> float_bytes(tensor_bytes.begin() + i, 
                                       tensor_bytes.begin() + i + sizeof(float));
        result.push_back(bytes_to_float(float_bytes));
    }
    
    return result;
}

// Serialize RL4SysAction to protobuf Action
rl4sys::Action serialize_action(const RL4SysAction& action) {
    rl4sys::Action action_proto;
    
    // Serialize observation
    if (!action.getObservation().empty()) {
        // Convert double vector to float vector
        std::vector<float> obs_float(action.getObservation().begin(), action.getObservation().end());
        auto obs_bytes = serialize_tensor(obs_float);
        action_proto.set_obs(obs_bytes.data(), obs_bytes.size());
    }
    
    // Serialize action value
    auto action_bytes = serialize_tensor({static_cast<float>(action.getActionValue())});
    action_proto.set_action(action_bytes.data(), action_bytes.size());
    
    // Serialize reward
    if (action.is_reward_set()) {
        auto reward_bytes = serialize_tensor({static_cast<float>(action.getReward().value())});
        action_proto.set_reward(reward_bytes.data(), reward_bytes.size());
    }
    
    // Set done flag
    action_proto.set_done(action.is_done());
    
    // Serialize extra data
    auto data = action.getData();
    for (const auto& [key, value] : data) {
        std::vector<float> value_vec = {static_cast<float>(std::stof(value))};
        auto value_bytes = serialize_tensor(value_vec);
        (*action_proto.mutable_extra_data())[key] = std::string(value_bytes.begin(), value_bytes.end());
    }
    
    return action_proto;
}

// Deserialize protobuf Action to RL4SysAction
RL4SysAction deserialize_action(const rl4sys::Action& action_proto) {
    // Deserialize observation
    std::vector<double> obs;
    if (!action_proto.obs().empty()) {
        std::vector<uint8_t> obs_bytes(action_proto.obs().begin(), action_proto.obs().end());
        auto obs_float = deserialize_tensor(obs_bytes);
        // Convert float vector to double vector
        obs.assign(obs_float.begin(), obs_float.end());
    }
    
    // Deserialize action
    std::vector<float> action_vec;
    if (!action_proto.action().empty()) {
        std::vector<uint8_t> action_bytes(action_proto.action().begin(), action_proto.action().end());
        action_vec = deserialize_tensor(action_bytes);
    }
    int64_t action_value = action_vec.empty() ? 0 : static_cast<int64_t>(action_vec[0]);
    
    // Deserialize reward
    std::optional<double> reward;
    if (!action_proto.reward().empty()) {
        std::vector<uint8_t> reward_bytes(action_proto.reward().begin(), action_proto.reward().end());
        auto reward_vec = deserialize_tensor(reward_bytes);
        if (!reward_vec.empty()) {
            reward = reward_vec[0];
        }
    }
    
    // Deserialize extra data
    std::map<std::string, std::string> data;
    for (const auto& [key, value] : action_proto.extra_data()) {
        std::vector<uint8_t> value_bytes(value.begin(), value.end());
        auto value_vec = deserialize_tensor(value_bytes);
        if (!value_vec.empty()) {
            data[key] = std::to_string(value_vec[0]);
        }
    }
    
    // Create and return RL4SysAction
    RL4SysAction result(obs, action_value, reward.value_or(0.0), action_proto.done(), data, 0);
    return result;
}

// Structured Logger implementation
StructuredLogger::StructuredLogger(const std::string& name, bool debug) 
    : logger_name_(name), debug_mode_(debug) {
    // Initialize spdlog logger
    logger_ = spdlog::get(name);
    if (!logger_) {
        logger_ = spdlog::stdout_color_mt(name);
    }
    logger_->set_level(debug ? spdlog::level::debug : spdlog::level::info);
}

void StructuredLogger::info(const std::string& msg, const std::map<std::string, std::string>& context) {
    std::stringstream ss;
    ss << msg << " - ";
    for (const auto& [key, value] : context) {
        ss << key << "=" << value << " ";
    }
    logger_->info(ss.str());
}

void StructuredLogger::debug(const std::string& msg, const std::map<std::string, std::string>& context) {
    if (debug_mode_) {
        std::stringstream ss;
        ss << msg << " - ";
        for (const auto& [key, value] : context) {
            ss << key << "=" << value << " ";
        }
        logger_->debug(ss.str());
    }
}

void StructuredLogger::error(const std::string& msg, const std::map<std::string, std::string>& context) {
    std::stringstream ss;
    ss << msg << " - ";
    for (const auto& [key, value] : context) {
        ss << key << "=" << value << " ";
    }
    logger_->error(ss.str());
}

} // namespace cppclient
} // namespace rl4sys 