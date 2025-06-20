#include "rl4sys_types.h"
#include <cstring>
#include <stdexcept>

namespace rl4sys {
namespace cppclient {

// RL4SysAction Implementation
RL4SysAction::RL4SysAction() 
    : reward_(0.0), done_(false), version_(0) {}

RL4SysAction::RL4SysAction(const std::vector<uint8_t>& obs_data,
                           const std::vector<uint8_t>& action_data,
                           double reward_value,
                           bool is_done,
                           int32_t version)
    : obs_bytes_(obs_data), action_bytes_(action_data), 
      reward_(reward_value), done_(is_done), version_(version) {}

void RL4SysAction::addExtraData(const std::string& key, const std::vector<uint8_t>& data) {
    extra_data_[key] = data;
}

void RL4SysAction::clear() {
    obs_bytes_.clear();
    action_bytes_.clear();
    extra_data_.clear();
    reward_ = 0.0;
    done_ = false;
    version_ = 0;
}

// RL4SysTrajectory Implementation
RL4SysTrajectory::RL4SysTrajectory() 
    : version_(0), completed_(false) {}

RL4SysTrajectory::RL4SysTrajectory(int32_t version) 
    : version_(version), completed_(false) {}

void RL4SysTrajectory::addAction(const RL4SysAction& action) {
    actions_.push_back(action);
}

void RL4SysTrajectory::addAction(RL4SysAction&& action) {
    actions_.push_back(std::move(action));
}

void RL4SysTrajectory::clear() {
    actions_.clear();
    completed_ = false;
}

// Utility functions implementation
namespace utils {
    
std::vector<uint8_t> serializeFloat(float value) {
    std::vector<uint8_t> result(sizeof(float));
    std::memcpy(result.data(), &value, sizeof(float));
    return result;
}

std::vector<uint8_t> serializeFloatVector(const std::vector<float>& values) {
    std::vector<uint8_t> result(values.size() * sizeof(float));
    std::memcpy(result.data(), values.data(), values.size() * sizeof(float));
    return result;
}

float deserializeFloat(const std::vector<uint8_t>& data) {
    if (data.size() != sizeof(float)) {
        throw std::invalid_argument("Invalid data size for float deserialization");
    }
    float value;
    std::memcpy(&value, data.data(), sizeof(float));
    return value;
}

std::vector<float> deserializeFloatVector(const std::vector<uint8_t>& data) {
    if (data.size() % sizeof(float) != 0) {
        throw std::invalid_argument("Invalid data size for float vector deserialization");
    }
    
    size_t count = data.size() / sizeof(float);
    std::vector<float> result(count);
    std::memcpy(result.data(), data.data(), data.size());
    return result;
}

std::vector<uint8_t> serializeAction(int64_t action) {
    std::vector<uint8_t> result(sizeof(int64_t));
    std::memcpy(result.data(), &action, sizeof(int64_t));
    return result;
}

int64_t deserializeAction(const std::vector<uint8_t>& data) {
    if (data.size() != sizeof(int64_t)) {
        throw std::invalid_argument("Invalid data size for action deserialization");
    }
    int64_t action;
    std::memcpy(&action, data.data(), sizeof(int64_t));
    return action;
}

} // namespace utils

} // namespace cppclient
} // namespace rl4sys