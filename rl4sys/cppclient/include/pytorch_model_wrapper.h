#pragma once

#include "rl4sys_types.h"
#include "logger.h"

#ifdef USE_PYTORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <optional>

namespace rl4sys {
namespace cppclient {

/**
 * @brief Simple model interface for local inference.
 * 
 * This provides a simplified interface for running action inference
 * without requiring full PyTorch integration in C++.
 */
class SimpleModel {
public:
    virtual ~SimpleModel() = default;
    
    /**
     * @brief Run inference to get an action for the given observation.
     * 
     * @param observation Input observation vector
     * @return Pair of (action_data, extra_data) where action_data is the
     *         serialized action and extra_data contains algorithm-specific
     *         information like log probabilities for PPO
     */
    virtual std::pair<std::vector<uint8_t>, std::map<std::string, std::vector<uint8_t>>>
    predict(const std::vector<float>& observation) = 0;
    
    /**
     * @brief Update the model weights from serialized state.
     * 
     * @param modelState Compressed model state bytes from server
     * @param isDiff Whether this is a differential update
     * @return True if update was successful
     */
    virtual bool updateWeights(const std::vector<uint8_t>& modelState, bool isDiff) = 0;
    
    /**
     * @brief Get model name for logging.
     */
    virtual std::string getModelName() const = 0;
};

#ifdef USE_PYTORCH

/**
 * @brief Base class for PyTorch-based models in C++.
 * 
 * This class provides the interface for loading PyTorch models from
 * compressed state dictionaries sent by the Python server and running
 * inference to generate actions.
 */
class PyTorchModelBase : public SimpleModel {
public:
    /**
     * @brief Constructor for PyTorch model base.
     * @param config Agent configuration containing algorithm parameters
     * @param logger Logger instance for debugging
     */
    PyTorchModelBase(const AgentConfig& config, std::shared_ptr<Logger> logger);
    
    virtual ~PyTorchModelBase() = default;
    
    // SimpleModel interface implementation
    std::pair<std::vector<uint8_t>, std::map<std::string, std::vector<uint8_t>>>
    predict(const std::vector<float>& observation) override;
    
    bool updateWeights(const std::vector<uint8_t>& modelState, bool isDiff) override;
    
    std::string getModelName() const override { return getAlgorithmName() + "_PyTorch"; }

protected:
    /**
     * @brief Create the algorithm-specific model architecture.
     * This method should initialize the model components.
     */
    virtual void createModel() = 0;
    
    /**
     * @brief Get all model parameters for weight updates.
     * @return Dictionary mapping parameter names to tensors
     */
    virtual torch::OrderedDict<std::string, torch::Tensor> named_parameters() = 0;
    
    /**
     * @brief Run algorithm-specific inference.
     * @param obs Input observation tensor
     * @return Pair of (action_tensor, extra_data_tensors)
     */
    virtual std::pair<torch::Tensor, std::map<std::string, torch::Tensor>>
    runInference(const torch::Tensor& obs) = 0;
    
    /**
     * @brief Get algorithm name for logging.
     */
    virtual std::string getAlgorithmName() const = 0;
    
    // Utility methods
    torch::Tensor deserializeTensor(const std::vector<uint8_t>& data);
    std::vector<uint8_t> serializeTensor(const torch::Tensor& tensor);
    
    // Model state management
    std::map<std::string, torch::Tensor> decompressStateDict(const std::vector<uint8_t>& compressed);
    void loadStateDict(const std::map<std::string, torch::Tensor>& state_dict);
    
    // Member variables
        AgentConfig config_;
    std::shared_ptr<Logger> logger_;
    torch::Device device_;
    bool model_initialized_;

private:
};

/**
 * @brief PPO model implementation using PyTorch C++.
 * 
 * Recreates the RLActorCritic architecture from the Python implementation:
 * - Actor network: obs -> action logits
 * - Critic network: obs -> state value
 */
class PPOModel : public PyTorchModelBase {
public:
    PPOModel(const AgentConfig& config, std::shared_ptr<Logger> logger);
    
protected:
    void createModel() override;
    torch::OrderedDict<std::string, torch::Tensor> named_parameters() override;
    std::pair<torch::Tensor, std::map<std::string, torch::Tensor>>
    runInference(const torch::Tensor& obs) override;
    std::string getAlgorithmName() const override { return "PPO"; }
    
private:
    int32_t input_size_;
    int32_t act_dim_;
    
    // Actor and critic networks
    torch::nn::Sequential actor_{nullptr};
    torch::nn::Sequential critic_{nullptr};
};

/**
 * @brief DQN model implementation using PyTorch C++.
 * 
 * Recreates the DeepQNetwork architecture from the Python implementation:
 * - Q-network: obs -> Q-values for each action
 * - Epsilon-greedy action selection
 */
class DQNModel : public PyTorchModelBase {
public:
    DQNModel(const AgentConfig& config, std::shared_ptr<Logger> logger);
    
protected:
    void createModel() override;
    torch::OrderedDict<std::string, torch::Tensor> named_parameters() override;
    std::pair<torch::Tensor, std::map<std::string, torch::Tensor>>
    runInference(const torch::Tensor& obs) override;
    std::string getAlgorithmName() const override { return "DQN"; }
    
private:
    int32_t input_size_;
    int32_t act_dim_;
    float epsilon_;
    float epsilon_min_;
    float epsilon_decay_;
    
    // Q-network
    torch::nn::Sequential q_network_{nullptr};
};

#endif // USE_PYTORCH

} // namespace cppclient
} // namespace rl4sys 