#include "pytorch_model_wrapper.h"

#ifdef USE_PYTORCH

namespace rl4sys {
namespace cppclient {

PPOModel::PPOModel(const AgentConfig& config, std::shared_ptr<Logger> logger)
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
    
    input_size_ = getParam("input_size");
    act_dim_ = getParam("act_dim");
    
    if (input_size_ <= 0 || act_dim_ <= 0) {
        throw std::runtime_error("Invalid PPO model parameters: input_size=" + 
                                std::to_string(input_size_) + ", act_dim=" + std::to_string(act_dim_));
    }
    
    logger_->info("PPO model parameters", 
                  "input_size", input_size_,
                  "act_dim", act_dim_);
    
    try {
        createModel();
        model_initialized_ = true;
        logger_->info("PyTorch PPO model created successfully");
    } catch (const std::exception& e) {
        logger_->error("Failed to create PyTorch PPO model", "error", e.what());
        model_initialized_ = false;
    }
}

void PPOModel::createModel() {
    logger_->info("Creating PPO actor-critic model architecture");
    
    // Create actor network - matches Python RLActor architecture
    // Python: mlp([input_size] + [32, 16, 8] + [act_dim], activation)
    actor_ = torch::nn::Sequential(
        torch::nn::Linear(input_size_, 32),
        torch::nn::ReLU(),
        torch::nn::Linear(32, 16),
        torch::nn::ReLU(),
        torch::nn::Linear(16, 8),
        torch::nn::ReLU(),
        torch::nn::Linear(8, act_dim_)
    );
    
    // Create critic network - matches Python RLCritic architecture  
    // Python: mlp([obs_dim] + [32, 16, 8] + [1], activation)
    critic_ = torch::nn::Sequential(
        torch::nn::Linear(input_size_, 32),
        torch::nn::ReLU(),
        torch::nn::Linear(32, 16), 
        torch::nn::ReLU(),
        torch::nn::Linear(16, 8),
        torch::nn::ReLU(),
        torch::nn::Linear(8, 1)
    );
    
    logger_->info("PPO model architecture created", 
                  "actor_params", actor_->parameters().size(),
                  "critic_params", critic_->parameters().size());
}

torch::OrderedDict<std::string, torch::Tensor> PPOModel::named_parameters() {
    torch::OrderedDict<std::string, torch::Tensor> params;
    
    // Add actor parameters with prefix
    auto actor_params = actor_->named_parameters();
    for (const auto& pair : actor_params) {
        params["actor." + pair.key()] = pair.value();
    }
    
    // Add critic parameters with prefix  
    auto critic_params = critic_->named_parameters();
    for (const auto& pair : critic_params) {
        params["critic." + pair.key()] = pair.value();
    }
    
    return params;
}

std::pair<torch::Tensor, std::map<std::string, torch::Tensor>>
PPOModel::runInference(const torch::Tensor& obs) {
    torch::NoGradGuard no_grad; // Disable gradient computation for inference
    
    try {
        // Get actor logits and critic value
        auto actor_logits = actor_->forward(obs);
        auto value = critic_->forward(obs);
        
        // Sample action from categorical distribution (matches Python implementation)
        auto probs = torch::softmax(actor_logits, -1);
        auto action = torch::multinomial(probs, 1);
        
        // Calculate log probability
        auto log_probs = torch::log_softmax(actor_logits, -1);
        auto log_prob = log_probs.gather(1, action);
        
        // Prepare extra data (matches Python RLActorCritic.step())
        std::map<std::string, torch::Tensor> extra_data;
        extra_data["v"] = value.squeeze(-1);  // Remove last dimension to match Python
        extra_data["logp_a"] = log_prob.squeeze(-1);  // Remove last dimension
        
        // Return action (remove batch and action dimensions for scalar action)
        auto action_scalar = action.squeeze(-1);
        
        logger_->debug("PPO inference completed", 
                       "action_shape", action_scalar.sizes(),
                       "value_shape", extra_data["v"].sizes(),
                       "logp_shape", extra_data["logp_a"].sizes());
        
        return {action_scalar, extra_data};
        
    } catch (const std::exception& e) {
        logger_->error("PPO inference failed", "error", e.what());
        throw;
    }
}

} // namespace cppclient
} // namespace rl4sys

#endif // USE_PYTORCH 