#pragma once

#include "rl4sys_types.h"
#include "logger.h"
#include "pytorch_model_wrapper.h"
#include <vector>
#include <memory>
#include <mutex>
#include <map>

namespace rl4sys {
namespace cppclient {



/**
 * @brief Simple random model for testing and fallback.
 * 
 * Generates random actions when a real model is not available.
 */
class RandomModel : public SimpleModel {
public:
    RandomModel(int32_t actionDim, double actionLimit = 1.0);
    
    std::pair<std::vector<uint8_t>, std::map<std::string, std::vector<uint8_t>>>
    predict(const std::vector<float>& observation) override;
    
    bool updateWeights(const std::vector<uint8_t>& modelState, bool isDiff) override;
    
    std::string getModelName() const override { return "RandomModel"; }

private:
    int32_t action_dim_;
    double action_limit_;
};

/**
 * @brief Manages local model storage and inference.
 * 
 * This class handles:
 * - Local model caching and version tracking
 * - Model updates from server (differential and full)
 * - Action inference using the local model
 * - Fallback to random actions when model is unavailable
 */
class ModelManager {
public:
    /**
     * @brief Construct a model manager.
     * 
     * @param config Agent configuration
     * @param logger Logger instance
     */
    ModelManager(const AgentConfig& config, std::shared_ptr<Logger> logger);
    
    /**
     * @brief Set the model from compressed state data.
     * 
     * @param modelState Compressed model state from server
     * @param version Model version
     * @param isDiff Whether this is a differential update
     * @return True if model was updated successfully
     */
    bool setModel(const std::vector<uint8_t>& modelState, int32_t version, bool isDiff);
    
    /**
     * @brief Generate an action for the given observation.
     * 
     * Uses the local model if available, otherwise falls back to random actions.
     * 
     * @param observation Input observation vector
     * @param version Model version to use for this action
     * @return RL4SysAction with generated action and extra data
     */
    RL4SysAction generateAction(const std::vector<float>& observation, int32_t version);
    
    /**
     * @brief Get the current model version.
     */
    int32_t getCurrentVersion() const;
    
    /**
     * @brief Check if a model is available for inference.
     */
    bool hasModel() const;
    
    /**
     * @brief Get algorithm name from configuration.
     */
    std::string getAlgorithmName() const { return config_.algorithm_name; }

private:
    /**
     * @brief Create a model instance based on algorithm configuration.
     */
    std::unique_ptr<SimpleModel> createModel();
    
#ifdef USE_PYTORCH
    /**
     * @brief Create a PyTorch model instance.
     */
    std::unique_ptr<PyTorchModelBase> createPyTorchModel();
#endif
    
    /**
     * @brief Decompress model state data.
     */
    std::vector<uint8_t> decompressModelState(const std::vector<uint8_t>& compressedData);
    
    AgentConfig config_;
    std::shared_ptr<Logger> logger_;
    
    mutable std::mutex model_mutex_;
    std::unique_ptr<SimpleModel> model_;
#ifdef USE_PYTORCH
    std::unique_ptr<PyTorchModelBase> pytorch_model_;
    bool use_pytorch_model_;
#endif
    std::unique_ptr<RandomModel> fallback_model_;
    int32_t current_version_;
    bool has_model_;
    
    // Algorithm parameters for model creation
    int32_t input_size_;
    int32_t action_dim_;
    double action_limit_;
};

} // namespace cppclient
} // namespace rl4sys