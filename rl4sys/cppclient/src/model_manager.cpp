#include "model_manager.h"
#include <random>
#include <algorithm>
#include <variant>

#ifdef USE_PYTORCH
#include "pytorch_model_wrapper.h"
#endif

namespace rl4sys {
namespace cppclient {

// RandomModel implementation
RandomModel::RandomModel(int32_t actionDim, double actionLimit) 
    : action_dim_(actionDim), action_limit_(actionLimit) {}

std::pair<std::vector<uint8_t>, std::map<std::string, std::vector<uint8_t>>>
RandomModel::predict(const std::vector<float>& observation) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-action_limit_, action_limit_);
    
    // Generate random action
    std::vector<float> action(action_dim_);
    for (int i = 0; i < action_dim_; ++i) {
        action[i] = dis(gen);
    }
    
    // For discrete action spaces (like DQN), convert to integer
    std::vector<uint8_t> action_data;
    if (action_dim_ == 1) {
        // Discrete action space - choose random action index
        std::uniform_int_distribution<int> int_dis(0, static_cast<int>(action_limit_) - 1);
        int64_t discrete_action = int_dis(gen);
        action_data = utils::serializeAction(discrete_action);
    } else {
        // Continuous action space
        action_data = utils::serializeFloatVector(action);
    }
    
    // Generate dummy extra data
    std::map<std::string, std::vector<uint8_t>> extra_data;
    extra_data["v"] = utils::serializeFloat(0.0f);  // Dummy value function
    extra_data["logp_a"] = utils::serializeFloat(-1.0f);  // Dummy log probability
    
    return {action_data, extra_data};
}

bool RandomModel::updateWeights(const std::vector<uint8_t>& modelState, bool isDiff) {
    // Random model doesn't have weights to update
    return true;
}

// ModelManager implementation
ModelManager::ModelManager(const AgentConfig& config, std::shared_ptr<Logger> logger)
    : config_(config), logger_(logger), current_version_(0), has_model_(false) 
#ifdef USE_PYTORCH
    , use_pytorch_model_(false)
#endif
{
    
    // Extract algorithm parameters
    auto getParam = [&config](const std::string& key) -> int32_t {
        auto it = config.algorithm_parameters.find(key);
        if (it != config.algorithm_parameters.end()) {
            if (std::holds_alternative<int32_t>(it->second)) {
                return std::get<int32_t>(it->second);
            }
        }
        return 0;
    };
    
    auto getDoubleParam = [&config](const std::string& key) -> double {
        auto it = config.algorithm_parameters.find(key);
        if (it != config.algorithm_parameters.end()) {
            if (std::holds_alternative<double>(it->second)) {
                return std::get<double>(it->second);
            }
        }
        return 0.0;
    };
    
    input_size_ = getParam("input_size");
    action_dim_ = getParam("act_dim");
    action_limit_ = config.act_limit;
    
    logger_->info("ModelManager initialized", 
                  "algorithm", config_.algorithm_name,
                  "input_size", input_size_,
                  "action_dim", action_dim_,
                  "action_limit", action_limit_);
    
    // Create initial models
    fallback_model_ = std::make_unique<RandomModel>(action_dim_, action_limit_);
    model_ = createModel();
}

bool ModelManager::setModel(const std::vector<uint8_t>& modelState, int32_t version, bool isDiff) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    if (modelState.empty()) {
        logger_->debug("Received empty model state - no update needed", "version", version);
        return true;
    }
    
    logger_->info("Updating model", 
                  "version", version,
                  "current_version", current_version_,
                  "is_diff", isDiff,
                  "model_size", modelState.size());
    
#ifdef USE_PYTORCH
    // Try PyTorch model first
    if (!pytorch_model_) {
        pytorch_model_ = createPyTorchModel();
        if (pytorch_model_) {
            logger_->info("Created PyTorch model", "algorithm", config_.algorithm_name);
        }
    }
    
    bool success = false;
    if (pytorch_model_) {
        success = pytorch_model_->updateWeights(modelState, isDiff);
        if (success) {
            use_pytorch_model_ = true;
            logger_->info("PyTorch model updated successfully", "version", version);
        } else {
            logger_->warning("PyTorch model update failed, falling back to random model");
            use_pytorch_model_ = false;
            success = fallback_model_->updateWeights(modelState, isDiff);
        }
    } else {
        logger_->info("Using fallback random model");
        use_pytorch_model_ = false;
        success = fallback_model_->updateWeights(modelState, isDiff);
    }
#else
    // No PyTorch support, use fallback model
    logger_->info("Using fallback random model (PyTorch not available)");
    bool success = fallback_model_->updateWeights(modelState, isDiff);
#endif
    
    if (success) {
        current_version_ = version;
        has_model_ = true;
        logger_->info("Model updated successfully", "new_version", version);
    } else {
        logger_->error("Failed to update model", "version", version);
    }
    
    return success;
}

RL4SysAction ModelManager::generateAction(const std::vector<float>& observation, int32_t version) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    if (observation.size() != static_cast<size_t>(input_size_)) {
        logger_->warning("Observation size mismatch", 
                        "expected", input_size_,
                        "got", observation.size());
    }
    
    // Choose which model to use
    SimpleModel* active_model = nullptr;
    std::string model_type = "fallback";
    
#ifdef USE_PYTORCH
    if (use_pytorch_model_ && pytorch_model_) {
        active_model = pytorch_model_.get();
        model_type = "pytorch";
    } else {
        active_model = fallback_model_.get();
    }
#else
    active_model = fallback_model_.get();
#endif
    
    if (!active_model) {
        logger_->error("No model available for inference");
        return RL4SysAction();
    }
    
    // Run inference
    auto [action_data, extra_data] = active_model->predict(observation);
    
    // Serialize observation
    std::vector<uint8_t> obs_data = utils::serializeFloatVector(observation);
    
    // Create action with no reward initially
    RL4SysAction action(obs_data, action_data, 0.0, false, version);
    
    // Add extra data
    for (const auto& [key, data] : extra_data) {
        action.addExtraData(key, data);
    }
    
    logger_->debug("Generated action", 
                   "model_type", model_type,
                   "model_version", version,
                   "obs_size", observation.size(),
                   "action_size", action_data.size(),
                   "extra_data_count", extra_data.size());
    
    return action;
}

int32_t ModelManager::getCurrentVersion() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return current_version_;
}

bool ModelManager::hasModel() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return has_model_;
}

std::unique_ptr<SimpleModel> ModelManager::createModel() {
#ifdef USE_PYTORCH
    // Try to create PyTorch model first
    auto pytorch_model = createPyTorchModel();
    if (pytorch_model) {
        logger_->info("Created PyTorch model", "algorithm", config_.algorithm_name);
        use_pytorch_model_ = true;
        return std::move(pytorch_model);
    }
#endif
    
    // Fallback to random model
    logger_->info("Creating fallback model", "type", "RandomModel");
    return std::make_unique<RandomModel>(action_dim_, action_limit_);
}

#ifdef USE_PYTORCH
std::unique_ptr<PyTorchModelBase> ModelManager::createPyTorchModel() {
    try {
        if (config_.algorithm_name == "PPO") {
            return std::make_unique<PPOModel>(config_, logger_);
        } else if (config_.algorithm_name == "DQN") {
            return std::make_unique<DQNModel>(config_, logger_);
        } else {
            logger_->warning("Unsupported algorithm for PyTorch model", 
                            "algorithm", config_.algorithm_name);
            return nullptr;
        }
    } catch (const std::exception& e) {
        logger_->error("Failed to create PyTorch model", 
                       "algorithm", config_.algorithm_name,
                       "error", e.what());
        return nullptr;
    }
}
#endif

std::vector<uint8_t> ModelManager::decompressModelState(const std::vector<uint8_t>& compressedData) {
    // TODO: Implement decompression (zlib/gzip)
    // For now, assume data is not compressed
    return compressedData;
}

} // namespace cppclient
} // namespace rl4sys