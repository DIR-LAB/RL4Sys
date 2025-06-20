#include "logger.h"
#include <chrono>
#include <iomanip>

namespace rl4sys {
namespace cppclient {

Logger::Logger(const std::string& name, LogLevel level) 
    : name_(name), level_(level) {}

void Logger::log(LogLevel level, const std::string& message, const std::string& context) {
    if (level < level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << getCurrentTimestamp() 
              << " - " << name_ 
              << " - " << levelToString(level) 
              << " - " << message;
              
    if (!context.empty() && context != "{}") {
        std::cout << " - " << context;
    }
    
    std::cout << std::endl;
}

std::string Logger::levelToString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:    return "DEBUG";
        case LogLevel::INFO:     return "INFO";
        case LogLevel::WARNING:  return "WARNING";
        case LogLevel::ERROR:    return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default:                 return "UNKNOWN";
    }
}

std::string Logger::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

// Template helper functions are implemented in the header file

} // namespace cppclient
} // namespace rl4sys