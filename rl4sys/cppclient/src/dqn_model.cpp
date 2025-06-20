#include "pytorch_model_wrapper.h"

#ifdef USE_PYTORCH

namespace rl4sys {
namespace cppclient {

DQNModel::DQNModel(const AgentConfig& config, std::shared_ptr<Logger> logger)
    : PyTorchModelBase(config, logger) {
    
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
    
    auto getFloatParam = [&config](const std::string& key, float default_val) -> float {
        auto it = config.algorithm_parameters.find(key);
        if (it != config.algorithm_parameters.end()) {
            if (std::holds_alternative<double>(it->second)) {
                return static_cast<float>(std::get<double>(it->second));
            }
        }
        return default_val;
    };
    
    input_size_ = getParam("input_size");
    act_dim_ = getParam("act_dim");
    epsilon_ = getFloatParam("epsilon", 1.0f);
    epsilon_min_ = getFloatParam("epsilon_min", 0.01f);
    epsilon_decay_ = getFloatParam("epsilon_decay", 5e-4f);
    
    if (input_size_ <= 0 || act_dim_ <= 0) {
        throw std::runtime_error("Invalid DQN model parameters: input_size=" + 
                                std::to_string(input_size_) + ", act_dim=" + std::to_string(act_dim_));
    }
    
    logger_->info("DQN model parameters", 
                  "input_size", input_size_,
                  "act_dim", act_dim_,
                  "epsilon", epsilon_,
                  "epsilon_min", epsilon_min_,
                  "epsilon_decay", epsilon_decay_);
    
    try {
        createModel();
        model_initialized_ = true;
        logger_->info("PyTorch DQN model created successfully");
    } catch (const std::exception& e) {
        logger_->error("Failed to create PyTorch DQN model", "error", e.what());
        model_initialized_ = false;
    }
}

void DQNModel::createModel() {
    logger_->info("Creating DQN network architecture");
    
    // Create Q-network - matches Python DeepQNetwork architecture
    // Python: nn.Sequential(
    //     nn.Linear(input_size, 64),
    //     nn.ReLU(),
    //     nn.Linear(64, 64),
    //     nn.ReLU(),
    //     nn.Linear(64, act_dim)
    // )
    q_network_ = torch::nn::Sequential(
        torch::nn::Linear(input_size_, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, act_dim_)
    );
    
    logger_->info("DQN network architecture created", 
                  "q_network_params", q_network_->parameters().size());
}

torch::OrderedDict<std::string, torch::Tensor> DQNModel::named_parameters() {
    torch::OrderedDict<std::string, torch::Tensor> params;
    
    // Add Q-network parameters with prefix
    auto q_params = q_network_->named_parameters();
    for (const auto& pair : q_params) {
        params["q_network." + pair.key()] = pair.value();
    }
    
    return params;
}

std::pair<torch::Tensor, std::map<std::string, torch::Tensor>>
DQNModel::runInference(const torch::Tensor& obs) {
    torch::NoGradGuard no_grad; // Disable gradient computation for inference
    
    try {
        // Get Q-values for all actions
        auto q_values = q_network_->forward(obs);
        
        // Epsilon-greedy action selection (matches Python implementation)
        torch::Tensor action;
        auto rand_val = static_cast<float>(rand()) / RAND_MAX;
        
        if (rand_val <= epsilon_) {
            // Random action (exploration)
            action = torch::randint(0, act_dim_, {1}, torch::kLong);
            logger_->debug("DQN random action", "epsilon", epsilon_, "rand_val", rand_val);
        } else {
            // Greedy action (exploitation)
            action = q_values.argmax(-1, true);
            logger_->debug("DQN greedy action", "max_q", q_values.max().item<float>());
        }
        
        // Decay epsilon (matches Python implementation)
        epsilon_ = std::max(epsilon_ * epsilon_decay_, epsilon_min_);
        
        // Prepare extra data (matches Python DeepQNetwork.step())
        std::map<std::string, torch::Tensor> extra_data;
        extra_data["q_val"] = q_values.squeeze(0); // Remove batch dimension
        extra_data["epsilon"] = torch::tensor(epsilon_);
        
        // Return action (remove batch dimension for scalar action)
        auto action_scalar = action.squeeze(0);
        
        logger_->debug("DQN inference completed", 
                       "action", action_scalar.item<int64_t>(),
                       "q_values_shape", extra_data["q_val"].sizes(),
                       "epsilon", epsilon_);
        
        return {action_scalar, extra_data};
        
    } catch (const std::exception& e) {
        logger_->error("DQN inference failed", "error", e.what());
        throw;
    }
}

} // namespace cppclient
} // namespace rl4sys

#endif // USE_PYTORCH 