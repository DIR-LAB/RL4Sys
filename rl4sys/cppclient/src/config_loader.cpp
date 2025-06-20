#include "config_loader.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

using json = nlohmann::json;

namespace rl4sys {
namespace cppclient {

AgentConfig ConfigLoader::loadFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw ConfigurationError("Cannot open configuration file: " + filePath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    return loadFromString(buffer.str());
}

AgentConfig ConfigLoader::loadFromString(const std::string& jsonContent) {
    try {
        json j = json::parse(jsonContent);
        
        AgentConfig config;
        
        // Core identification (required)
        config.client_id = getStringValue(&j, "client_id", "default_client", true);
        config.algorithm_name = getStringValue(&j, "algorithm_name", "PPO", true);
        config.algorithm_type = getStringValue(&j, "type", "onpolicy", false);
        
        // Server connection (required)
        config.train_server_address = getStringValue(&j, "train_server_address", "localhost:50051", true);
        
        // Algorithm parameters (required)
        if (j.contains("algorithm_parameters") && j["algorithm_parameters"].is_object()) {
            config.algorithm_parameters = parseAlgorithmParameters(&j["algorithm_parameters"]);
        } else {
            throw ConfigurationError("Missing required 'algorithm_parameters' section");
        }
        
        // Client behavior settings (optional with defaults)
        config.act_limit = getDoubleValue(&j, "act_limit", 1.0, false);
        config.max_traj_length = getIntValue(&j, "max_traj_length", 1000, false);
        config.send_frequency = getIntValue(&j, "send_frequency", 10, false);
        
        // Connection settings (optional with defaults)
        config.connection_timeout_seconds = getIntValue(&j, "connection_timeout_seconds", 10, false);
        config.request_timeout_seconds = getIntValue(&j, "request_timeout_seconds", 30, false);
        config.enable_compression = getBoolValue(&j, "enable_compression", true, false);
        
        // Debug settings (optional with defaults)
        config.debug = getBoolValue(&j, "debug", false, false);
        config.log_level = getStringValue(&j, "log_level", "INFO", false);
        
        validateConfig(config);
        return config;
        
    } catch (const json::parse_error& e) {
        throw ConfigurationError("JSON parse error: " + std::string(e.what()));
    } catch (const json::type_error& e) {
        throw ConfigurationError("JSON type error: " + std::string(e.what()));
    }
}

void ConfigLoader::validateConfig(const AgentConfig& config) {
    // Validate required fields
    if (config.client_id.empty()) {
        throw ConfigurationError("client_id cannot be empty");
    }
    
    if (config.algorithm_name.empty()) {
        throw ConfigurationError("algorithm_name cannot be empty");
    }
    
    if (config.train_server_address.empty()) {
        throw ConfigurationError("train_server_address cannot be empty");
    }
    
    // Validate algorithm type
    if (config.algorithm_type != "onpolicy" && config.algorithm_type != "offpolicy") {
        throw ConfigurationError("algorithm type must be 'onpolicy' or 'offpolicy', got: " + config.algorithm_type);
    }
    
    // Validate ranges
    if (config.send_frequency <= 0) {
        throw ConfigurationError("send_frequency must be positive, got: " + std::to_string(config.send_frequency));
    }
    
    if (config.max_traj_length <= 0) {
        throw ConfigurationError("max_traj_length must be positive, got: " + std::to_string(config.max_traj_length));
    }
    
    if (config.connection_timeout_seconds <= 0) {
        throw ConfigurationError("connection_timeout_seconds must be positive, got: " + std::to_string(config.connection_timeout_seconds));
    }
    
    if (config.request_timeout_seconds <= 0) {
        throw ConfigurationError("request_timeout_seconds must be positive, got: " + std::to_string(config.request_timeout_seconds));
    }
    
    // Validate required algorithm parameters
    auto checkParam = [&config](const std::string& paramName) {
        if (config.algorithm_parameters.find(paramName) == config.algorithm_parameters.end()) {
            throw ConfigurationError("Missing required algorithm parameter: " + paramName);
        }
    };
    
    // Common required parameters for all algorithms
    checkParam("input_size");
    checkParam("act_dim");
    
    // Algorithm-specific validation
    if (config.algorithm_name == "PPO") {
        checkParam("batch_size");
        checkParam("traj_per_epoch");
        checkParam("gamma");
        checkParam("lam");
    } else if (config.algorithm_name == "DQN") {
        checkParam("buffer_size");
        checkParam("learning_rate");
        checkParam("gamma");
    }
}

void ConfigLoader::saveToFile(const AgentConfig& config, const std::string& filePath) {
    std::string jsonStr = configToJsonString(config);
    
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw ConfigurationError("Cannot open file for writing: " + filePath);
    }
    
    file << jsonStr;
    file.close();
}

std::string ConfigLoader::configToJsonString(const AgentConfig& config) {
    json j;
    
    // Core identification
    j["client_id"] = config.client_id;
    j["algorithm_name"] = config.algorithm_name;
    j["type"] = config.algorithm_type;
    
    // Server connection
    j["train_server_address"] = config.train_server_address;
    
    // Algorithm parameters
    json algParams;
    for (const auto& [key, value] : config.algorithm_parameters) {
        std::visit([&algParams, &key](const auto& v) {
            algParams[key] = v;
        }, value);
    }
    j["algorithm_parameters"] = algParams;
    
    // Client behavior settings
    j["act_limit"] = config.act_limit;
    j["max_traj_length"] = config.max_traj_length;
    j["send_frequency"] = config.send_frequency;
    
    // Connection settings
    j["connection_timeout_seconds"] = config.connection_timeout_seconds;
    j["request_timeout_seconds"] = config.request_timeout_seconds;
    j["enable_compression"] = config.enable_compression;
    
    // Debug settings
    j["debug"] = config.debug;
    j["log_level"] = config.log_level;
    
    return j.dump(4); // Pretty print with 4-space indentation
}

std::map<std::string, std::variant<int32_t, double, std::string, bool>>
ConfigLoader::parseAlgorithmParameters(const void* jsonParams) {
    const json* j = static_cast<const json*>(jsonParams);
    std::map<std::string, std::variant<int32_t, double, std::string, bool>> params;
    
    for (const auto& [key, value] : j->items()) {
        if (value.is_null()) {
            // Skip null values (optional parameters)
            continue;
        } else if (value.is_boolean()) {
            params[key] = value.get<bool>();
        } else if (value.is_string()) {
            params[key] = value.get<std::string>();
        } else if (value.is_number_integer()) {
            params[key] = value.get<int32_t>();
        } else if (value.is_number_float()) {
            params[key] = value.get<double>();
        } else {
            throw ConfigurationError("Unsupported parameter type for key '" + key + "'. Only int, double, string, bool, and null are supported.");
        }
    }
    
    return params;
}

std::string ConfigLoader::getStringValue(const void* jsonPtr, const std::string& key, 
                                       const std::string& defaultValue, bool required) {
    const json* j = static_cast<const json*>(jsonPtr);
    
    if (!j->contains(key)) {
        if (required) {
            throw ConfigurationError("Missing required field: " + key);
        }
        return defaultValue;
    }
    
    if (!(*j)[key].is_string()) {
        throw ConfigurationError("Field '" + key + "' must be a string");
    }
    
    return (*j)[key].get<std::string>();
}

int32_t ConfigLoader::getIntValue(const void* jsonPtr, const std::string& key, 
                                 int32_t defaultValue, bool required) {
    const json* j = static_cast<const json*>(jsonPtr);
    
    if (!j->contains(key)) {
        if (required) {
            throw ConfigurationError("Missing required field: " + key);
        }
        return defaultValue;
    }
    
    if (!(*j)[key].is_number_integer()) {
        throw ConfigurationError("Field '" + key + "' must be an integer");
    }
    
    return (*j)[key].get<int32_t>();
}

double ConfigLoader::getDoubleValue(const void* jsonPtr, const std::string& key, 
                                   double defaultValue, bool required) {
    const json* j = static_cast<const json*>(jsonPtr);
    
    if (!j->contains(key)) {
        if (required) {
            throw ConfigurationError("Missing required field: " + key);
        }
        return defaultValue;
    }
    
    if (!(*j)[key].is_number()) {
        throw ConfigurationError("Field '" + key + "' must be a number");
    }
    
    return (*j)[key].get<double>();
}

bool ConfigLoader::getBoolValue(const void* jsonPtr, const std::string& key, 
                               bool defaultValue, bool required) {
    const json* j = static_cast<const json*>(jsonPtr);
    
    if (!j->contains(key)) {
        if (required) {
            throw ConfigurationError("Missing required field: " + key);
        }
        return defaultValue;
    }
    
    if (!(*j)[key].is_boolean()) {
        throw ConfigurationError("Field '" + key + "' must be a boolean");
    }
    
    return (*j)[key].get<bool>();
}

} // namespace cppclient
} // namespace rl4sys