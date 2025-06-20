#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <memory>
#include <mutex>

namespace rl4sys {
namespace cppclient {

/**
 * @brief Simple structured logger for the C++ client.
 * 
 * Provides basic logging functionality similar to the Python client's
 * StructuredLogger, with support for different log levels and structured
 * context information.
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class Logger {
public:
    /**
     * @brief Construct a logger with the given name and level.
     * 
     * @param name Logger name (e.g., "RL4SysAgent")
     * @param level Minimum log level to output
     */
    explicit Logger(const std::string& name, LogLevel level = LogLevel::INFO);
    
    /**
     * @brief Log a debug message.
     * 
     * @param message The log message
     * @param context Optional context information as key-value pairs
     */
    template<typename... Args>
    void debug(const std::string& message, Args&&... context);
    
    /**
     * @brief Log an info message.
     * 
     * @param message The log message
     * @param context Optional context information as key-value pairs
     */
    template<typename... Args>
    void info(const std::string& message, Args&&... context);
    
    /**
     * @brief Log a warning message.
     * 
     * @param message The log message
     * @param context Optional context information as key-value pairs
     */
    template<typename... Args>
    void warning(const std::string& message, Args&&... context);
    
    /**
     * @brief Log an error message.
     * 
     * @param message The log message
     * @param context Optional context information as key-value pairs
     */
    template<typename... Args>
    void error(const std::string& message, Args&&... context);
    
    /**
     * @brief Log a critical message.
     * 
     * @param message The log message
     * @param context Optional context information as key-value pairs
     */
    template<typename... Args>
    void critical(const std::string& message, Args&&... context);
    
    /**
     * @brief Set the minimum log level.
     */
    void setLevel(LogLevel level) { level_ = level; }
    
    /**
     * @brief Get the current log level.
     */
    LogLevel getLevel() const { return level_; }
    
    /**
     * @brief Check if debug logging is enabled.
     */
    bool isDebugEnabled() const { return level_ <= LogLevel::DEBUG; }
    
    /**
     * @brief Check if info logging is enabled.
     */
    bool isInfoEnabled() const { return level_ <= LogLevel::INFO; }

private:
    /**
     * @brief Internal logging function.
     */
    void log(LogLevel level, const std::string& message, const std::string& context = "");
    
    /**
     * @brief Format context information into a string.
     */
    template<typename... Args>
    std::string formatContext(Args&&... args);
    
    /**
     * @brief Get string representation of log level.
     */
    std::string levelToString(LogLevel level) const;
    
    /**
     * @brief Get current timestamp string.
     */
    std::string getCurrentTimestamp() const;
    
    std::string name_;
    LogLevel level_;
    mutable std::mutex mutex_;
};

// Template implementations

template<typename... Args>
void Logger::debug(const std::string& message, Args&&... context) {
    if (level_ <= LogLevel::DEBUG) {
        log(LogLevel::DEBUG, message, formatContext(std::forward<Args>(context)...));
    }
}

template<typename... Args>
void Logger::info(const std::string& message, Args&&... context) {
    if (level_ <= LogLevel::INFO) {
        log(LogLevel::INFO, message, formatContext(std::forward<Args>(context)...));
    }
}

template<typename... Args>
void Logger::warning(const std::string& message, Args&&... context) {
    if (level_ <= LogLevel::WARNING) {
        log(LogLevel::WARNING, message, formatContext(std::forward<Args>(context)...));
    }
}

template<typename... Args>
void Logger::error(const std::string& message, Args&&... context) {
    if (level_ <= LogLevel::ERROR) {
        log(LogLevel::ERROR, message, formatContext(std::forward<Args>(context)...));
    }
}

template<typename... Args>
void Logger::critical(const std::string& message, Args&&... context) {
    if (level_ <= LogLevel::CRITICAL) {
        log(LogLevel::CRITICAL, message, formatContext(std::forward<Args>(context)...));
    }
}

template<typename... Args>
std::string Logger::formatContext(Args&&... args) {
    if (sizeof...(args) == 0) {
        return "{}";
    }
    
    std::stringstream ss;
    ss << "{";
    
    // Simple comma-separated format for now
    bool first = true;
    ((ss << (first ? "" : ", ") << args, first = false), ...);
    
    ss << "}";
    return ss.str();
}

} // namespace cppclient
} // namespace rl4sys