#pragma once

#include <vector>
#include <string>
#include <map>
#include <optional>
#include <memory>
#include "rl4sys_agent.h"
#include "rl4sys.pb.h"
#include <spdlog/spdlog.h>

namespace rl4sys {
namespace cppclient {

// Helper functions for float/byte conversion
std::vector<uint8_t> float_to_bytes(float value);
float bytes_to_float(const std::vector<uint8_t>& bytes);

// Tensor serialization/deserialization
std::vector<uint8_t> serialize_tensor(const std::vector<float>& tensor);
std::vector<float> deserialize_tensor(const std::vector<uint8_t>& tensor_bytes);

// Action serialization/deserialization
rl4sys_proto::Action serialize_action(const RL4SysAction& action);
RL4SysAction deserialize_action(const rl4sys_proto::Action& action_proto);

// Structured Logger class
class StructuredLogger {
public:
    StructuredLogger(const std::string& name, bool debug = false);
    
    void info(const std::string& msg, const std::map<std::string, std::string>& context = {});
    void debug(const std::string& msg, const std::map<std::string, std::string>& context = {});
    void error(const std::string& msg, const std::map<std::string, std::string>& context = {});

private:
    std::string logger_name_;
    bool debug_mode_;
    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace cppclient
} // namespace rl4sys 