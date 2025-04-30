// Placeholder for rl4sys/cppclient/src/config_loader.cpp
#include "config_loader.h"
#include <fstream>      // For std::ifstream
#include <stdexcept>    // For std::runtime_error
#include <iostream>     // For error messages (replace with logger ideally)

// --- JSON Library ---
// You need to include your chosen JSON library here.
// Example using nlohmann/json:
#include <nlohmann/json.hpp> // Assuming nlohmann/json is available
using json = nlohmann::json;
// --------------------


namespace rl4sys {
namespace cppclient {

AgentConfig ConfigLoader::loadFromFile(const std::string& filePath) {
    std::ifstream configFileStream(filePath);
    if (!configFileStream.is_open()) {
        throw std::runtime_error("Could not open config file: " + filePath);
    }

    json configJson;
    try {
        configFileStream >> configJson;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("Failed to parse JSON config file: " + std::string(e.what()));
    }

    AgentConfig config;

    // Helper lambda to get required field or throw
    auto get_required = [&](const std::string& key) {
        if (!configJson.contains(key)) {
            throw std::runtime_error("Missing required config field: " + key + " in " + filePath);
        }
        return configJson.at(key);
    };

    // Helper lambda to get optional field or use default
    auto get_optional = [&](const std::string& key, const auto& defaultValue) {
        return configJson.value(key, defaultValue);
    };

    try {
        config.clientId = get_required("client_id").get<std::string>();
        config.trainServerAddress = get_required("train_server_address").get<std::string>();
        config.sendFrequency = get_optional("send_frequency", config.sendFrequency); // Use default if missing
        config.actLimit = get_optional("act_limit", config.actLimit);

        // --- Load other parameters as needed ---
        // Example: loading nested algorithm parameters
        // if (configJson.contains("algorithm_parameters")) {
        //     const auto& algoParams = configJson.at("algorithm_parameters");
        //     config.learningRate = algoParams.value("pi_lr", 0.0003); // Example
        // }

    } catch (const json::type_error& e) {
         throw std::runtime_error("JSON type error in config file: " + std::string(e.what()));
    } catch (const std::runtime_error& e) {
        throw; // Re-throw missing field errors
    }


    return config;
}

} // namespace cppclient
} // namespace rl4sys