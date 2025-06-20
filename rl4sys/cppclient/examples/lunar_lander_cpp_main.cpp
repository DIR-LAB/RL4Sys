/**
 * @file lunar_lander_cpp_main.cpp
 * @brief Example C++ client for Lunar Lander environment using RL4Sys.
 * 
 * This example demonstrates how to use the C++ RL4SysAgent to interact
 * with the RL4Sys server for reinforcement learning training.
 * 
 * It provides feature parity with the Python lunar_lander.py example,
 * including:
 * - Configuration loading from JSON
 * - gRPC communication with RL4Sys server
 * - Real PyTorch model inference (when available)
 * - Asynchronous trajectory sending
 * - Comprehensive logging and statistics
 */

#include "rl4sys_agent.h"
#include "config_loader.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>  // For std::clamp (C++17)
#include <iomanip>
#include <cmath>

using namespace rl4sys::cppclient;

/**
 * @brief Lunar Lander environment simulation.
 * 
 * This simulates the OpenAI Gym LunarLander-v3 environment behavior
 * to demonstrate the C++ RL4Sys client capabilities.
 * 
 * State space: 8-dimensional continuous
 * Action space: 4 discrete actions (0: do nothing, 1: left engine, 2: main engine, 3: right engine)
 */
class LunarLanderEnv {
public:
    /**
     * @brief Constructor with configurable parameters.
     * 
     * @param seed Random seed for reproducibility
     * @param max_steps Maximum steps per episode
     */
    LunarLanderEnv(int seed = 42, int max_steps = 1000) 
        : rng_(seed), episode_steps_(0), max_steps_(max_steps), total_episodes_(0) {
        reset();
    }
    
    /**
     * @brief Reset the environment for a new episode.
     * 
     * @return Initial observation (8-dimensional state vector)
     */
    std::vector<float> reset() {
        episode_steps_ = 0;
        total_episodes_++;
        done_ = false;
        
        // Initialize Lunar Lander state (similar to Gym environment)
        // State: [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
        std::uniform_real_distribution<float> pos_dist(-0.5f, 0.5f);
        std::uniform_real_distribution<float> vel_dist(-0.1f, 0.1f);
        std::uniform_real_distribution<float> angle_dist(-0.2f, 0.2f);
        
        observation_.resize(8);
        observation_[0] = pos_dist(rng_);      // x position
        observation_[1] = 1.0f + pos_dist(rng_); // y position (start high)
        observation_[2] = vel_dist(rng_);      // x velocity
        observation_[3] = vel_dist(rng_);      // y velocity
        observation_[4] = angle_dist(rng_);    // angle
        observation_[5] = vel_dist(rng_);      // angular velocity
        observation_[6] = 0.0f;                // left leg contact
        observation_[7] = 0.0f;                // right leg contact
        
        return observation_;
    }
    
    /**
     * @brief Take a step in the environment.
     * 
     * @param action Action to take (0-3 for discrete Lunar Lander)
     * @return Tuple of (next_observation, reward, done)
     */
    std::tuple<std::vector<float>, float, bool> step(int action) {
        episode_steps_++;
        
        // Clamp action to valid range
        action = std::clamp(action, 0, 3);
        
        // Physics simulation (simplified)
        std::uniform_real_distribution<float> noise_dist(-0.01f, 0.01f);
        
        // Apply action effects
        float thrust_x = 0.0f, thrust_y = 0.0f;
        switch (action) {
            case 0: // Do nothing
                break;
            case 1: // Left engine
                thrust_x = -0.05f;
                break;
            case 2: // Main engine
                thrust_y = 0.1f;
                break;
            case 3: // Right engine
                thrust_x = 0.05f;
                break;
        }
        
        // Update physics
        observation_[2] += thrust_x + noise_dist(rng_); // x velocity
        observation_[3] += thrust_y - 0.02f + noise_dist(rng_); // y velocity (gravity)
        observation_[0] += observation_[2] * 0.1f; // x position
        observation_[1] += observation_[3] * 0.1f; // y position
        observation_[4] += observation_[5] * 0.1f + noise_dist(rng_); // angle
        observation_[5] += noise_dist(rng_); // angular velocity
        
        // Clamp values to reasonable ranges
        observation_[0] = std::clamp(observation_[0], -2.0f, 2.0f);
        observation_[1] = std::clamp(observation_[1], 0.0f, 2.0f);
        observation_[2] = std::clamp(observation_[2], -1.0f, 1.0f);
        observation_[3] = std::clamp(observation_[3], -1.0f, 1.0f);
        observation_[4] = std::clamp(observation_[4], -1.0f, 1.0f);
        observation_[5] = std::clamp(observation_[5], -1.0f, 1.0f);
        
        // Check for landing (ground contact)
        bool landed = observation_[1] <= 0.1f;
        if (landed) {
            observation_[6] = 1.0f; // left leg contact
            observation_[7] = 1.0f; // right leg contact
        }
        
        // Calculate reward (similar to Gym LunarLander)
        float reward = 0.0f;
        
        // Distance from target (center, ground)
        float distance_penalty = -(std::abs(observation_[0]) + std::abs(observation_[1]));
        reward += distance_penalty * 0.1f;
        
        // Velocity penalty
        float velocity_penalty = -(std::abs(observation_[2]) + std::abs(observation_[3]));
        reward += velocity_penalty * 0.1f;
        
        // Angle penalty
        reward -= std::abs(observation_[4]) * 0.1f;
        
        // Action penalty (fuel cost)
        if (action != 0) {
            reward -= 0.1f;
        }
        
        // Landing reward/penalty
        if (landed) {
            if (std::abs(observation_[0]) < 0.2f && std::abs(observation_[2]) < 0.2f && 
                std::abs(observation_[3]) < 0.2f && std::abs(observation_[4]) < 0.2f) {
                // Successful landing
                reward += 200.0f;
            } else {
                // Crash landing
                reward -= 100.0f;
            }
            done_ = true;
        }
        
        // Check episode termination conditions
        if (episode_steps_ >= max_steps_ || observation_[1] > 2.0f || std::abs(observation_[0]) > 2.0f) {
            done_ = true;
            if (!landed) {
                reward -= 100.0f; // Crashed or flew away
            }
        }
        
        return {observation_, reward, done_};
    }
    
    bool isDone() const { return done_; }
    int getEpisodeSteps() const { return episode_steps_; }
    int getTotalEpisodes() const { return total_episodes_; }
    
    /**
     * @brief Check if the episode ended successfully (good landing).
     */
    bool isSuccessfulLanding() const {
        return done_ && observation_[6] > 0.0f && observation_[7] > 0.0f &&
               std::abs(observation_[0]) < 0.2f && std::abs(observation_[2]) < 0.2f &&
               std::abs(observation_[3]) < 0.2f && std::abs(observation_[4]) < 0.2f;
    }

private:
    std::mt19937 rng_;
    std::vector<float> observation_;
    int episode_steps_;
    int max_steps_;
    int total_episodes_;
    bool done_;
};

/**
 * @brief Statistics tracking for simulation performance.
 */
struct SimulationStats {
    int total_episodes = 0;
    int total_steps = 0;
    int successful_landings = 0;
    int crashes = 0;
    float total_reward = 0.0f;
    std::vector<float> episode_rewards;
    std::vector<int> episode_lengths;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    
    void startTimer() {
        start_time = std::chrono::steady_clock::now();
    }
    
    void recordEpisode(float reward, int length, bool success) {
        total_episodes++;
        total_steps += length;
        total_reward += reward;
        episode_rewards.push_back(reward);
        episode_lengths.push_back(length);
        
        if (success) {
            successful_landings++;
        } else {
            crashes++;
        }
    }
    
    void printSummary() const {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        float avg_reward = total_episodes > 0 ? total_reward / total_episodes : 0.0f;
        float avg_length = total_episodes > 0 ? static_cast<float>(total_steps) / total_episodes : 0.0f;
        float success_rate = total_episodes > 0 ? static_cast<float>(successful_landings) / total_episodes * 100.0f : 0.0f;
        
        std::cout << "\n=== Simulation Summary ===" << std::endl;
        std::cout << "Total episodes: " << total_episodes << std::endl;
        std::cout << "Total steps: " << total_steps << std::endl;
        std::cout << "Training time: " << duration.count() << " seconds" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Average reward: " << avg_reward << std::endl;
        std::cout << "Average episode length: " << avg_length << std::endl;
        std::cout << "Successful landings: " << successful_landings << " (" << success_rate << "%)" << std::endl;
        std::cout << "Crashes: " << crashes << std::endl;
        
        if (!episode_rewards.empty()) {
            auto minmax = std::minmax_element(episode_rewards.begin(), episode_rewards.end());
            std::cout << "Best episode reward: " << *minmax.second << std::endl;
            std::cout << "Worst episode reward: " << *minmax.first << std::endl;
        }
    }
};

/**
 * @brief Run a single episode of Lunar Lander.
 * 
 * @param agent RL4Sys agent for action generation
 * @param env Lunar Lander environment
 * @return Total episode reward
 */
float runEpisode(RL4SysAgent& agent, LunarLanderEnv& env) {
    auto observation = env.reset();
    float total_reward = 0.0f;
    int step_count = 0;
    
    std::shared_ptr<RL4SysTrajectory> trajectory = nullptr;
    
    while (!env.isDone()) {
        // Get action from agent (this uses real PyTorch model when available)
        auto result = agent.requestForAction(trajectory, observation);
        if (!result) {
            std::cerr << "❌ Failed to get action from agent" << std::endl;
            break;
        }
        
        trajectory = result->first;
        RL4SysAction action = result->second;
        
        // Extract discrete action value (0-3)
        auto action_data = action.getAction();
        int action_value = 0; // Default: do nothing
        if (!action_data.empty()) {
            // Convert action data to integer
            action_value = static_cast<int>(action_data[0]) % 4;
        }
        
        // Take step in environment
        auto [next_obs, reward, done] = env.step(action_value);
        
        // Update action with reward
        agent.updateActionReward(action, reward);
        
        // Add action to trajectory
        agent.addToTrajectory(trajectory, action);
        
        total_reward += reward;
        step_count++;
        
        // Progress indicator for long episodes
        if (step_count % 50 == 0) {
            std::cout << "  Step " << step_count << ", Action: " << action_value 
                      << ", Reward: " << std::fixed << std::setprecision(1) << reward 
                      << ", Total: " << std::fixed << std::setprecision(1) << total_reward << std::endl;
        }
        
        // Check if episode is done
        if (done) {
            // Mark the end of trajectory and send to server
            agent.markEndOfTrajectory(trajectory, action);
            break;
        }
        
        observation = next_obs;
    }
    
    return total_reward;
}

/**
 * @brief Main function demonstrating C++ RL4Sys client usage.
 */
int main(int argc, char* argv[]) {
    std::cout << "🚀 RL4Sys C++ Client - Lunar Lander Example" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Parse command line arguments
    std::string config_path = "../../../rl4sys/examples/lunar/luna_conf.json";  // Correct path from build directory
    bool debug = false;
    int num_episodes = 10;
    int max_steps = 1000;
    int seed = 42;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--episodes" && i + 1 < argc) {
            num_episodes = std::atoi(argv[++i]);
        } else if (arg == "--max-steps" && i + 1 < argc) {
            max_steps = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --config <path>     Configuration file path (default: ../../../rl4sys/examples/lunar/luna_conf.json)" << std::endl;
            std::cout << "  --debug             Enable debug logging" << std::endl;
            std::cout << "  --episodes <n>      Number of episodes to run (default: 10)" << std::endl;
            std::cout << "  --max-steps <n>     Maximum steps per episode (default: 1000)" << std::endl;
            std::cout << "  --seed <n>          Random seed (default: 42)" << std::endl;
            std::cout << "  --help              Show this help message" << std::endl;
            return 0;
        }
    }
    
    try {
        // Initialize RL4Sys agent
        std::cout << "🔧 Initializing RL4Sys agent..." << std::endl;
        std::cout << "   Config: " << config_path << std::endl;
        std::cout << "   Debug: " << (debug ? "ON" : "OFF") << std::endl;
        
        RL4SysAgent agent(config_path, debug);
        
        std::cout << "✅ Agent initialized successfully!" << std::endl;
        std::cout << "📊 Current model version: " << agent.getCurrentModelVersion() << std::endl;
        
        // Initialize environment
        LunarLanderEnv env(seed, max_steps);
        std::cout << "🌙 Lunar Lander environment ready (seed: " << seed << ")" << std::endl;
        
        // Initialize statistics
        SimulationStats stats;
        stats.startTimer();
        
        std::cout << "\n🎮 Starting training episodes..." << std::endl;
        std::cout << "Episodes: " << num_episodes << ", Max steps: " << max_steps << std::endl;
        std::cout << "==========================================" << std::endl;
        
        // Run episodes
        for (int episode = 0; episode < num_episodes; ++episode) {
            std::cout << "\n--- Episode " << (episode + 1) << "/" << num_episodes << " ---" << std::endl;
            
            float reward = runEpisode(agent, env);
            bool success = env.isSuccessfulLanding();
            int length = env.getEpisodeSteps();
            
            stats.recordEpisode(reward, length, success);
            
            std::cout << "🎯 Episode " << (episode + 1) << " completed:" << std::endl;
            std::cout << "   Reward: " << std::fixed << std::setprecision(1) << reward << std::endl;
            std::cout << "   Steps: " << length << std::endl;
            std::cout << "   Result: " << (success ? "✅ Successful Landing!" : "❌ Crash/Failure") << std::endl;
            std::cout << "   Model version: " << agent.getCurrentModelVersion() << std::endl;
            
            // Small delay between episodes for server processing
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Print final statistics
        stats.printSummary();
        
        // Flush any remaining trajectories
        std::cout << "\n📤 Flushing remaining trajectories..." << std::endl;
        agent.flushTrajectories();
        
        // Close agent
        std::cout << "🔒 Closing agent connection..." << std::endl;
        agent.close();
        
        std::cout << "\n🎉 Training completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}