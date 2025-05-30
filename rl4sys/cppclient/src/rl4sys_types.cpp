#include "rl4sys_types.h"

namespace rl4sys {
namespace cppclient {

// --- RL4SysAction Implementation ---

RL4SysAction::RL4SysAction() : actionValue(0), actionReward(std::nullopt), done(false), version(0) {}

RL4SysAction::RL4SysAction(const std::vector<double>& obs, int64_t act, double reward, bool done, 
                           const std::map<std::string, std::string>& data, int ver)
    : observation(obs), actionValue(act), actionReward(reward), done(done), extraData(data), version(ver) {}

void RL4SysAction::updateReward(double reward) {
    actionReward = reward;
}

int64_t RL4SysAction::getActionValue() const {
    return actionValue;
}

void RL4SysAction::setActionValue(int64_t value) {
    actionValue = value;
}

std::optional<double> RL4SysAction::getReward() const {
    return actionReward;
}

const std::vector<double>& RL4SysAction::getObservation() const {
    return observation;
}

bool RL4SysAction::is_reward_set() const {
    return actionReward.has_value();
}

bool RL4SysAction::is_done() const {
    return done;
}

void RL4SysAction::set_done(bool is_done) {
    done = is_done;
}

const std::map<std::string, std::string>& RL4SysAction::getData() const {
    return extraData;
}

int RL4SysAction::getVersion() const {
    return version;
}

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

const std::vector<std::vector<double>>& RL4SysTrajectory::getObservations() const {
    return observations;
}

const std::vector<RL4SysAction>& RL4SysTrajectory::getActions() const {
    return actions;
}

} // namespace cppclient
} // namespace rl4sys 