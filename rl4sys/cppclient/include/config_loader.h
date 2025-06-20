#pragma once

#include "rl4sys_types.h"
#include <string>
#include <stdexcept>

namespace rl4sys {
namespace cppclient {

/**
 * @brief Exception thrown when configuration loading fails.
 */
class ConfigurationError : public std::runtime_error {
public:
    explicit ConfigurationError(const std::string& message) 
        : std::runtime_error("Configuration Error: " + message) {}
};

/**
 * @brief Utility class for loading agent configuration from JSON files.
 * 
 * This class provides comprehensive JSON configuration loading that matches
 * the Python client's configuration system, supporting all fields from
 * luna_conf.json and other configuration files.
 */
class ConfigLoader {
public:
    /**
     * @brief Load agent configuration from a JSON file.
     * 
     * Supports the full configuration structure used by the Python client:
     * - client_id: Unique identifier for this client
     * - algorithm_name: Name of the RL algorithm (PPO, DQN, etc.)
     * - algorithm_parameters: Algorithm-specific hyperparameters
     * - type: Algorithm type ("onpolicy" or "offpolicy")
     * - train_server_address: gRPC server address
     * - send_frequency: Number of trajectories to buffer before sending
     * - act_limit: Action space limit
     * - max_traj_length: Maximum trajectory length
     * 
     * @param filePath Path to the JSON configuration file
     * @return AgentConfig structure with parsed configuration
     * @throws ConfigurationError if file cannot be read or parsed
     * @throws ConfigurationError if required fields are missing
     */
    static AgentConfig loadFromFile(const std::string& filePath);
    
    /**
     * @brief Load configuration from a JSON string.
     * 
     * @param jsonContent JSON string content
     * @return AgentConfig structure with parsed configuration
     * @throws ConfigurationError if JSON is invalid or required fields are missing
     */
    static AgentConfig loadFromString(const std::string& jsonContent);
    
    /**
     * @brief Validate that all required fields are present in the configuration.
     * 
     * @param config Configuration to validate
     * @throws ConfigurationError if validation fails
     */
    static void validateConfig(const AgentConfig& config);
    
    /**
     * @brief Save configuration to a JSON file.
     * 
     * @param config Configuration to save
     * @param filePath Output file path
     * @throws ConfigurationError if file cannot be written
     */
    static void saveToFile(const AgentConfig& config, const std::string& filePath);
    
    /**
     * @brief Convert configuration to JSON string.
     * 
     * @param config Configuration to convert
     * @return JSON string representation
     */
    static std::string configToJsonString(const AgentConfig& config);

private:
    /**
     * @brief Parse algorithm parameters from JSON object.
     * 
     * Handles conversion of JSON values to appropriate C++ types:
     * - Numbers -> int32_t or double
     * - Strings -> std::string  
     * - Booleans -> bool
     * - null -> no entry (optional parameters)
     * 
     * @param jsonParams JSON object containing parameters
     * @return Map of parameter name to variant value
     */
    static std::map<std::string, std::variant<int32_t, double, std::string, bool>>
    parseAlgorithmParameters(const void* jsonParams);
    
    /**
     * @brief Convert variant parameter value to JSON.
     * 
     * @param value Variant parameter value
     * @return JSON representation of the value
     */
    static void* variantToJson(const std::variant<int32_t, double, std::string, bool>& value);
    
    /**
     * @brief Get string value from JSON object with validation.
     * 
     * @param json JSON object
     * @param key Key to extract
     * @param defaultValue Default value if key is missing
     * @param required Whether the field is required
     * @return String value
     * @throws ConfigurationError if required field is missing
     */
    static std::string getStringValue(const void* jsonPtr, const std::string& key, 
                                     const std::string& defaultValue = "", 
                                     bool required = false);
    
    /**
     * @brief Get integer value from JSON object with validation.
     * 
     * @param jsonPtr JSON object pointer
     * @param key Key to extract
     * @param defaultValue Default value if key is missing
     * @param required Whether the field is required
     * @return Integer value
     * @throws ConfigurationError if required field is missing or invalid type
     */
    static int32_t getIntValue(const void* jsonPtr, const std::string& key, 
                              int32_t defaultValue = 0, 
                              bool required = false);
    
    /**
     * @brief Get double value from JSON object with validation.
     * 
     * @param jsonPtr JSON object pointer
     * @param key Key to extract
     * @param defaultValue Default value if key is missing
     * @param required Whether the field is required
     * @return Double value
     * @throws ConfigurationError if required field is missing or invalid type
     */
    static double getDoubleValue(const void* jsonPtr, const std::string& key, 
                                double defaultValue = 0.0, 
                                bool required = false);
    
    /**
     * @brief Get boolean value from JSON object with validation.
     * 
     * @param jsonPtr JSON object pointer
     * @param key Key to extract
     * @param defaultValue Default value if key is missing
     * @param required Whether the field is required
     * @return Boolean value
     * @throws ConfigurationError if required field is missing or invalid type
     */
    static bool getBoolValue(const void* jsonPtr, const std::string& key, 
                            bool defaultValue = false, 
                            bool required = false);
};

} // namespace cppclient
} // namespace rl4sys