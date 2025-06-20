#include "rl4sys_agent.h"
#include "config_loader.h"
#include "logger.h"
#include "model_manager.h"
#include "rl4sys_stub_wrapper.h"

// gRPC and protobuf includes
#include <grpcpp/grpcpp.h>
#include "rl4sys.pb.h"
#include "rl4sys.grpc.pb.h"

#include <chrono>
#include <thread>

namespace rl4sys {
namespace cppclient {

RL4SysAgent::RL4SysAgent(const std::string& configFilePath, bool debug) 
    : local_model_version_(0), stop_flag_(false), initialized_(false), connected_(false) {
    
    // Load configuration
    try {
        config_ = ConfigLoader::loadFromFile(configFilePath);
        if (debug) {
            config_.debug = true;
            config_.log_level = "DEBUG";
        }
    } catch (const ConfigurationError& e) {
        throw std::runtime_error("Failed to load configuration: " + std::string(e.what()));
    }
    
    // Initialize logger
    LogLevel logLevel = LogLevel::INFO;
    if (config_.log_level == "DEBUG") logLevel = LogLevel::DEBUG;
    else if (config_.log_level == "WARNING") logLevel = LogLevel::WARNING;
    else if (config_.log_level == "ERROR") logLevel = LogLevel::ERROR;
    else if (config_.log_level == "CRITICAL") logLevel = LogLevel::CRITICAL;
    
    logger_ = std::make_unique<Logger>("RL4SysAgent", logLevel);
    
    logger_->info("Initializing RL4SysAgent", 
                  "client_id", config_.client_id,
                  "algorithm", config_.algorithm_name,
                  "server_address", config_.train_server_address);
    
    // Initialize model manager
    model_manager_ = std::make_unique<ModelManager>(config_, std::shared_ptr<Logger>(logger_.get(), [](Logger*){}));
    
    // Create gRPC channel
    createGrpcChannel();
    
    // Initialize server algorithm
    initializeServerAlgorithm();
    
    // Get initial model
    if (!getModelFromServer(-1)) {
        logger_->warning("Failed to get initial model from server, using random fallback");
    }
    
    // Start background sending thread
    send_thread_ = std::make_unique<std::thread>(&RL4SysAgent::sendThreadWorker, this);
    
    initialized_ = true;
    logger_->info("RL4SysAgent initialization complete");
}

RL4SysAgent::~RL4SysAgent() {
    close();
}

void RL4SysAgent::createGrpcChannel() {
    // Configure channel options with compression
    grpc::ChannelArguments args;
    if (config_.enable_compression) {
        args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
    }
    
    // Set timeouts
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 30000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);
    args.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
    args.SetInt(GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS, 300000);
    
    // Create channel
    channel_ = grpc::CreateCustomChannel(
        config_.train_server_address,
        grpc::InsecureChannelCredentials(),  // Use secure credentials in production
        args
    );
    
    if (!channel_) {
        throw std::runtime_error("Failed to create gRPC channel to " + config_.train_server_address);
    }
    
    // Create stub wrapper
    stub_ = std::make_unique<RLServiceStubInterface>(channel_);
    
    // Test connectivity
    auto deadline = std::chrono::system_clock::now() + 
                   std::chrono::seconds(config_.connection_timeout_seconds);
    if (channel_->WaitForConnected(deadline)) {
        connected_ = true;
        logger_->info("Connected to RL4Sys server", "address", config_.train_server_address);
    } else {
        logger_->warning("Could not immediately connect to server, will retry on requests");
    }
}

void RL4SysAgent::initializeServerAlgorithm() {
    rl4sys::InitRequest request;
    rl4sys::InitResponse response;
    grpc::ClientContext context;
    
    // Set deadline
    auto deadline = std::chrono::system_clock::now() + 
                   std::chrono::seconds(config_.request_timeout_seconds);
    context.set_deadline(deadline);
    
    // Enable compression
    if (config_.enable_compression) {
        context.set_compression_algorithm(GRPC_COMPRESS_GZIP);
    }
    
    // Fill request
    request.set_client_id(config_.client_id);
    request.set_algorithm_name(config_.algorithm_name);
    
    // Convert algorithm parameters
    convertAlgorithmParameters(config_.algorithm_parameters, *request.mutable_algorithm_parameters());
    
    logger_->info("Initializing server algorithm", 
                  "client_id", config_.client_id,
                  "algorithm", config_.algorithm_name);
    
    // Make gRPC call
    grpc::Status status = stub_->InitAlgorithm(&context, request, &response);
    
    if (status.ok()) {
        if (response.is_success()) {
            logger_->info("Server algorithm initialized successfully", "message", response.message());
        } else {
            throw std::runtime_error("Server algorithm initialization failed: " + response.message());
        }
    } else {
        throw std::runtime_error("gRPC InitAlgorithm failed: " + status.error_message());
    }
}

bool RL4SysAgent::getModelFromServer(int32_t expectedVersion) {
    rl4sys::GetModelRequest request;
    rl4sys::ModelResponse response;
    grpc::ClientContext context;
    
    // Set deadline
    auto deadline = std::chrono::system_clock::now() + 
                   std::chrono::seconds(config_.request_timeout_seconds);
    context.set_deadline(deadline);
    
    // Enable compression
    if (config_.enable_compression) {
        context.set_compression_algorithm(GRPC_COMPRESS_GZIP);
    }
    
    // Fill request
    request.set_client_id(config_.client_id);
    request.set_client_version(local_model_version_);
    request.set_expected_version(expectedVersion);
    
    logger_->debug("Requesting model from server", 
                   "current_version", local_model_version_.load(),
                   "expected_version", expectedVersion);
    
    // Make gRPC call
    grpc::Status status = stub_->GetModel(&context, request, &response);
    
    if (!status.ok()) {
        logger_->error("gRPC GetModel failed", 
                       "error_code", status.error_code(),
                       "error_message", status.error_message());
        return false;
    }
    
    // Check if we got an empty response (already have latest version)
    if (response.model_state().empty()) {
        logger_->debug("Already have latest model version", "version", response.version());
        return true;
    }
    
    // Update model
    std::vector<uint8_t> modelState(response.model_state().begin(), response.model_state().end());
    bool success = model_manager_->setModel(modelState, response.version(), response.is_diff());
    
    if (success) {
        local_model_version_ = response.version();
        logger_->info("Model updated successfully", 
                      "version", response.version(),
                      "is_diff", response.is_diff(),
                      "model_size", modelState.size());
    }
    
    return success;
}

std::optional<std::pair<std::shared_ptr<RL4SysTrajectory>, RL4SysAction>>
RL4SysAgent::requestForAction(std::shared_ptr<RL4SysTrajectory> trajectory, 
                              const std::vector<float>& observation) {
    if (!initialized_) {
        logger_->error("Agent not initialized");
        return std::nullopt;
    }
    
    // Create new trajectory if needed
    if (!trajectory || trajectory->isCompleted()) {
        trajectory = std::make_shared<RL4SysTrajectory>(local_model_version_);
        logger_->debug("Created new trajectory", "version", local_model_version_.load());
    }
    
    // Generate action using local model
    RL4SysAction action = model_manager_->generateAction(observation, trajectory->getVersion());
    
    if (action.getObservation().empty()) {
        logger_->error("Failed to generate action");
        return std::nullopt;
    }
    
    logger_->debug("Generated action for trajectory", 
                   "trajectory_version", trajectory->getVersion(),
                   "action_size", action.getAction().size(),
                   "obs_size", observation.size());
    
    return std::make_pair(trajectory, action);
}

void RL4SysAgent::addToTrajectory(std::shared_ptr<RL4SysTrajectory> trajectory, 
                                  const RL4SysAction& action) {
    if (!trajectory) {
        logger_->error("Cannot add action to null trajectory");
        return;
    }
    
    trajectory->addAction(action);
    logger_->debug("Added action to trajectory", 
                   "trajectory_size", trajectory->size(),
                   "trajectory_version", trajectory->getVersion());
}

void RL4SysAgent::updateActionReward(RL4SysAction& action, double reward) {
    action.setReward(reward);
    logger_->debug("Updated action reward", "reward", reward);
}

bool RL4SysAgent::markEndOfTrajectory(std::shared_ptr<RL4SysTrajectory> trajectory, 
                                      RL4SysAction& lastAction) {
    if (!trajectory) {
        logger_->error("Cannot mark end of null trajectory");
        return false;
    }
    
    // Mark last action as done and add to trajectory
    lastAction.setDone(true);
    trajectory->addAction(lastAction);
    trajectory->markCompleted();
    
    logger_->info("Marked trajectory as completed", 
                  "trajectory_size", trajectory->size(),
                  "trajectory_version", trajectory->getVersion());
    
    // Add to send queue
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        send_queue_.push(trajectory);
    }
    send_cv_.notify_one();
    
    return true;
}

bool RL4SysAgent::flushTrajectories() {
    logger_->info("Flushing all buffered trajectories");
    
    std::vector<std::shared_ptr<RL4SysTrajectory>> trajectories;
    {
        std::lock_guard<std::mutex> lock(send_mutex_);
        while (!send_queue_.empty()) {
            trajectories.push_back(send_queue_.front());
            send_queue_.pop();
        }
    }
    
    if (trajectories.empty()) {
        logger_->debug("No trajectories to flush");
        return true;
    }
    
    return sendTrajectoriesToServer(trajectories);
}

void RL4SysAgent::sendThreadWorker() {
    logger_->info("Trajectory sending thread started");
    
    std::vector<std::shared_ptr<RL4SysTrajectory>> batch;
    
    while (!stop_flag_) {
        std::unique_lock<std::mutex> lock(send_mutex_);
        
        // Wait for trajectories or stop signal
        send_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
            return !send_queue_.empty() || stop_flag_;
        });
        
        if (stop_flag_) {
            break;
        }
        
        // Collect trajectories for batch sending
        while (!send_queue_.empty() && static_cast<int32_t>(batch.size()) < config_.send_frequency) {
            batch.push_back(send_queue_.front());
            send_queue_.pop();
        }
        
        lock.unlock();
        
        // Send batch if we have enough trajectories
        if (static_cast<int32_t>(batch.size()) >= config_.send_frequency) {
            logger_->info("Sending trajectory batch", "count", batch.size());
            
            if (sendTrajectoriesToServer(batch)) {
                logger_->debug("Trajectory batch sent successfully");
            } else {
                logger_->error("Failed to send trajectory batch");
            }
            
            batch.clear();
        }
    }
    
    // Send any remaining trajectories on shutdown
    if (!batch.empty()) {
        logger_->info("Sending final trajectory batch", "count", batch.size());
        sendTrajectoriesToServer(batch);
    }
    
    logger_->info("Trajectory sending thread stopped");
}

bool RL4SysAgent::sendTrajectoriesToServer(const std::vector<std::shared_ptr<RL4SysTrajectory>>& trajectories) {
    if (trajectories.empty()) {
        return true;
    }
    
    rl4sys::SendTrajectoriesRequest request;
    rl4sys::SendTrajectoriesResponse response;
    grpc::ClientContext context;
    
    // Set deadline
    auto deadline = std::chrono::system_clock::now() + 
                   std::chrono::seconds(config_.request_timeout_seconds);
    context.set_deadline(deadline);
    
    // Enable compression
    if (config_.enable_compression) {
        context.set_compression_algorithm(GRPC_COMPRESS_GZIP);
    }
    
    // Fill request
    request.set_client_id(config_.client_id);
    
    // Filter trajectories based on algorithm type
    std::vector<std::shared_ptr<RL4SysTrajectory>> validTrajectories;
    for (const auto& traj : trajectories) {
        if (config_.algorithm_type == "onpolicy") {
            // For on-policy algorithms, only send trajectories with matching version
            if (traj->getVersion() == local_model_version_ && traj->isValid()) {
                validTrajectories.push_back(traj);
            }
        } else {
            // For off-policy algorithms, send all valid trajectories
            if (traj->isValid()) {
                validTrajectories.push_back(traj);
            }
        }
    }
    
    if (validTrajectories.empty()) {
        logger_->debug("No valid trajectories to send after filtering");
        return true;
    }
    
    // Convert trajectories to protobuf
    for (const auto& traj : validTrajectories) {
        rl4sys::Trajectory* protoTraj = request.add_trajectories();
        convertToProtoTrajectory(*traj, protoTraj);
    }
    
    logger_->info("Sending trajectories to server", 
                  "total_count", trajectories.size(),
                  "valid_count", validTrajectories.size(),
                  "algorithm_type", config_.algorithm_type);
    
    // Make gRPC call
    grpc::Status status = stub_->SendTrajectories(&context, request, &response);
    
    if (!status.ok()) {
        logger_->error("gRPC SendTrajectories failed", 
                       "error_code", status.error_code(),
                       "error_message", status.error_message());
        return false;
    }
    
    // Check if model was updated
    if (response.model_updated()) {
        logger_->info("Server indicates model update available", 
                      "new_version", response.new_version(),
                      "current_version", local_model_version_.load());
        
        // Get updated model
        if (getModelFromServer(response.new_version())) {
            logger_->info("Model updated after trajectory sending");
        } else {
            logger_->warning("Failed to get updated model from server");
        }
    }
    
    return true;
}

void RL4SysAgent::convertToProtoAction(const RL4SysAction& action, rl4sys::Action* protoAction) {
    protoAction->set_obs(action.getObservation().data(), action.getObservation().size());
    protoAction->set_action(action.getAction().data(), action.getAction().size());
    
    // Serialize reward
    std::vector<uint8_t> rewardData = utils::serializeFloat(static_cast<float>(action.getReward()));
    protoAction->set_reward(rewardData.data(), rewardData.size());
    
    protoAction->set_done(action.isDone());
    
    // Add extra data
    for (const auto& [key, data] : action.getExtraData()) {
        (*protoAction->mutable_extra_data())[key] = std::string(data.begin(), data.end());
    }
}

void RL4SysAgent::convertToProtoTrajectory(const RL4SysTrajectory& trajectory, rl4sys::Trajectory* protoTrajectory) {
    protoTrajectory->set_version(trajectory.getVersion());
    
    for (const auto& action : trajectory.getActions()) {
        rl4sys::Action* protoAction = protoTrajectory->add_actions();
        convertToProtoAction(action, protoAction);
    }
}

void RL4SysAgent::convertAlgorithmParameters(
    const std::map<std::string, std::variant<int32_t, double, std::string, bool>>& params,
    google::protobuf::Map<std::string, rl4sys::ParameterValue>& protoParams) {
    
    for (const auto& [key, value] : params) {
        rl4sys::ParameterValue paramValue;
        
        std::visit([&paramValue](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, int32_t>) {
                paramValue.set_int_value(v);
            } else if constexpr (std::is_same_v<T, double>) {
                paramValue.set_float_value(v);
            } else if constexpr (std::is_same_v<T, std::string>) {
                paramValue.set_string_value(v);
            } else if constexpr (std::is_same_v<T, bool>) {
                paramValue.set_bool_value(v);
            }
        }, value);
        
        protoParams[key] = paramValue;
    }
}

int32_t RL4SysAgent::getCurrentModelVersion() const {
    return local_model_version_;
}

void RL4SysAgent::close() {
    if (!initialized_) {
        return;
    }
    
    logger_->info("Closing RL4SysAgent");
    
    // Stop background thread
    stop_flag_ = true;
    send_cv_.notify_all();
    
    if (send_thread_ && send_thread_->joinable()) {
        send_thread_->join();
    }
    
    // Flush remaining trajectories
    flushTrajectories();
    
    initialized_ = false;
    logger_->info("RL4SysAgent closed successfully");
}

} // namespace cppclient
} // namespace rl4sys