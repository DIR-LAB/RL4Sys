// Placeholder for rl4sys/cppclient/include/config_loader.h
#pragma once

#include "rl4sys_agent.h" // For AgentConfig struct
#include <string>

namespace rl4sys {
namespace cppclient {

/**
 * @brief Utility class for loading agent configuration from a JSON file.
 */
class ConfigLoader {
public:
    /**
     * @brief Loads agent configuration from the specified file path.
     * @param filePath The path to the JSON configuration file.
     * @return AgentConfig structure populated with values from the file.
     * @throws std::runtime_error if the file cannot be opened or parsed, or if required fields are missing.
     */
    static AgentConfig loadFromFile(const std::string& filePath);
};

} // namespace cppclient
} // namespace rl4sys