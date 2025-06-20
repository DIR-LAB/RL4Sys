#include "pytorch_model_wrapper.h"
#include "utils/compression.h"

#ifdef USE_PYTORCH

#include <sstream>
#include <iostream>

namespace rl4sys {
namespace cppclient {

PyTorchModelBase::PyTorchModelBase(const AgentConfig& config, std::shared_ptr<Logger> logger)
    : config_(config), logger_(logger), device_(torch::kCPU), model_initialized_(false) {
    
    logger_->info("Initializing PyTorch model base", 
                  "algorithm", config_.algorithm_name,
                  "device", device_.str());
    
    // Note: createModel() will be called by derived classes in their constructors
    // We can't call virtual functions in the base constructor
}

std::pair<std::vector<uint8_t>, std::map<std::string, std::vector<uint8_t>>>
PyTorchModelBase::predict(const std::vector<float>& observation) {
    if (!model_initialized_) {
        logger_->error("PyTorch model not initialized");
        return {{}, {}};
    }
    
    try {
        // Convert observation to tensor
        torch::Tensor obs_tensor = torch::from_blob(
            const_cast<float*>(observation.data()), 
            {static_cast<int64_t>(observation.size())}, 
            torch::kFloat
        ).to(device_);
        
        // Add batch dimension if needed
        if (obs_tensor.dim() == 1) {
            obs_tensor = obs_tensor.unsqueeze(0);
        }
        
        // Run inference
        auto [action_tensor, extra_tensors] = runInference(obs_tensor);
        
        // Serialize results
        auto action_data = serializeTensor(action_tensor);
        
        std::map<std::string, std::vector<uint8_t>> extra_data;
        for (const auto& [key, tensor] : extra_tensors) {
            extra_data[key] = serializeTensor(tensor);
        }
        
        logger_->debug("PyTorch inference completed", 
                       "obs_size", observation.size(),
                       "action_size", action_data.size(),
                       "extra_data_count", extra_data.size());
        
        return {action_data, extra_data};
        
    } catch (const std::exception& e) {
        logger_->error("PyTorch inference failed", "error", e.what());
        return {{}, {}};
    }
}

bool PyTorchModelBase::updateWeights(const std::vector<uint8_t>& modelState, bool isDiff) {
    if (!model_initialized_) {
        logger_->error("Cannot update weights - PyTorch model not initialized");
        return false;
    }
    
    if (modelState.empty()) {
        logger_->debug("Received empty model state - no update needed");
        return true;
    }
    
    try {
        // Decompress the state dict
        auto new_state = decompressStateDict(modelState);
        
        if (isDiff) {
            // Apply differential update
            auto current_dict = named_parameters();
            for (const auto& [name, tensor] : new_state) {
                auto param_tensor = current_dict.find(name);
                if (param_tensor != nullptr) {
                    param_tensor->copy_(tensor);
                    logger_->debug("Updated parameter", "name", name, "shape", tensor.sizes());
                }
            }
            logger_->info("Applied PyTorch model diff", "changed_params", new_state.size());
        } else {
            // Full model replacement
            loadStateDict(new_state);
            logger_->info("Loaded complete PyTorch model", "total_params", new_state.size());
        }
        
        return true;
    } catch (const std::exception& e) {
        logger_->error("Failed to update PyTorch model weights", "error", e.what());
        return false;
    }
}

torch::Tensor PyTorchModelBase::deserializeTensor(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        return torch::empty({0});
    }
    
    try {
        std::stringstream ss(std::string(data.begin(), data.end()));
        torch::serialize::InputArchive archive;
        archive.load_from(ss);
        
        torch::Tensor tensor;
        archive.read("tensor", tensor);
        return tensor.to(device_);
    } catch (const std::exception& e) {
        logger_->error("Failed to deserialize tensor", "error", e.what());
        return torch::empty({0});
    }
}

std::vector<uint8_t> PyTorchModelBase::serializeTensor(const torch::Tensor& tensor) {
    try {
        std::stringstream ss;
        torch::serialize::OutputArchive archive;
        archive.write("tensor", tensor.cpu()); // Always serialize on CPU
        archive.save_to(ss);
        
        std::string str = ss.str();
        return std::vector<uint8_t>(str.begin(), str.end());
    } catch (const std::exception& e) {
        logger_->error("Failed to serialize tensor", "error", e.what());
        return {};
    }
}

std::map<std::string, torch::Tensor> 
PyTorchModelBase::decompressStateDict(const std::vector<uint8_t>& compressed) {
    try {
        // First decompress using zlib (application-level compression)
        auto decompressed = rl4sys::utils::zlibDecompress(compressed);
        
        // Then deserialize PyTorch state dict
        std::stringstream ss(std::string(decompressed.begin(), decompressed.end()));
        torch::serialize::InputArchive archive;
        archive.load_from(ss);
        
        std::map<std::string, torch::Tensor> state_dict;
        
        // Note: This is a simplified approach. In practice, we'd need to iterate
        // through the keys in the archive. For now, we'll assume the Python server
        // sends the state dict in a specific format.
        
        // TODO: Implement proper state dict deserialization
        // This would require knowing the exact format used by the Python server
        
        logger_->debug("Decompressed state dict", "size", decompressed.size());
        return state_dict;
        
    } catch (const std::exception& e) {
        logger_->error("Failed to decompress state dict", "error", e.what());
        return {};
    }
}

void PyTorchModelBase::loadStateDict(const std::map<std::string, torch::Tensor>& state_dict) {
    try {
        auto current_dict = named_parameters();
        
        for (const auto& [name, tensor] : state_dict) {
            auto param_tensor = current_dict.find(name);
            if (param_tensor != nullptr) {
                param_tensor->copy_(tensor.to(device_));
                logger_->debug("Loaded parameter", "name", name, "shape", tensor.sizes());
            } else {
                logger_->warning("Unknown parameter in state dict", "name", name);
            }
        }
        
        logger_->info("State dict loaded successfully", "params_loaded", state_dict.size());
    } catch (const std::exception& e) {
        logger_->error("Failed to load state dict", "error", e.what());
        throw;
    }
}

} // namespace cppclient
} // namespace rl4sys

#endif // USE_PYTORCH 