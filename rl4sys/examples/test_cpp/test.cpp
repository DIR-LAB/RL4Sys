// Include RL4Sys C++ client headers
#include "rl4sys_agent.h"

#include <iostream>
#include <vector>
#include <optional>
#include <random>
#include <chrono>
#include <string>

// Simple stub environment mimicking observation/action/reward loop
class SimpleEnv {
public:
    SimpleEnv(int input_dim, int act_dim, int seed)
        : input_dim_(input_dim), act_dim_(act_dim), rng_(seed), dist_obs_(-1.0, 1.0), dist_reward_(-1.0, 1.0) {}

    std::vector<double> reset() {
        // Reset environment state and return initial observation
        step_count_ = 0;
        return sampleObservation();
    }

    struct StepResult {
        std::vector<double> next_obs;
        double reward;
        bool done;
    };

    StepResult step(int action, int max_steps) {
        // In a real environment, use the action. Here we just create dummy transition.
        (void)action; // suppress unused warning
        ++step_count_;
        StepResult result;
        result.next_obs = sampleObservation();
        result.reward = dist_reward_(rng_);
        result.done = (step_count_ >= max_steps); 
        return result;
    }

private:
    std::vector<double> sampleObservation() {
        std::vector<double> obs(input_dim_);
        for (double &v : obs) {
            v = dist_obs_(rng_);
        }
        return obs;
    }

    int input_dim_;
    int act_dim_;
    int step_count_ = 0;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_obs_;
    std::uniform_real_distribution<double> dist_reward_;
};

int main(int argc, char** argv) {
    // Default parameters
    int seed = 1;
    int num_iterations = 10;
    int max_moves = 200;
    std::string config_path = "./test_conf.json";

    // Very simple argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_val = [&]() -> std::string {
            if (i + 1 < argc) return argv[++i];
            throw std::runtime_error("Missing value for argument: " + arg);
        };

        if (arg == "--seed") {
            seed = std::stoi(next_val());
        } else if (arg == "--number-of-iterations") {
            num_iterations = std::stoi(next_val());
        } else if (arg == "--number-of-moves") {
            max_moves = std::stoi(next_val());
        } else if (arg == "--config") {
            config_path = next_val();
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    std::cout << "RL4Sys C++ Test Simulation\n";
    std::cout << "Seed: " << seed << ", Iterations: " << num_iterations << ", Max moves: " << max_moves << std::endl;
    std::cout << "Using config: " << config_path << std::endl;

    try {
        rl4sys::cppclient::RL4SysAgent agent(config_path);
        SimpleEnv env(/*input_dim=*/8, /*act_dim=*/4, seed);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> action_dist(0, 3); // fallback random action if server fails

        for (int iter = 0; iter < num_iterations; ++iter) {
            std::cout << "\nIteration " << iter << std::endl;

            auto obs = env.reset();
            std::optional<rl4sys::cppclient::RL4SysTrajectory> traj; // empty initially
            int moves = 0;
            bool done = false;
            double cumulative_reward = 0.0;
            auto start_time = std::chrono::steady_clock::now();

            while (!done && moves < max_moves) {
                // Request action from server
                auto resp_opt = agent.requestForAction(traj, obs);
                if (!resp_opt.has_value()) {
                    // Fallback to random action if server unavailable
                    if (!traj.has_value()) traj.emplace();
                    int random_act = action_dist(rng);
                    rl4sys::cppclient::RL4SysAction fallback_action;
                    fallback_action.updateReward(0.0);
                    fallback_action.setActionValue(random_act);
                    agent.addToTrajectory(*traj, fallback_action);

                    // Environment step
                    auto step_res = env.step(random_act, max_moves);
                    obs = step_res.next_obs;
                    cumulative_reward += step_res.reward;
                    done = step_res.done;
                    ++moves;
                    continue;
                }

                // Unpack trajectory & action
                auto [updated_traj, action] = resp_opt.value();
                traj = updated_traj; // store back
                int action_val = static_cast<int>(action.getActionValue());

                // Step environment
                auto step_res = env.step(action_val, max_moves);

                // Update reward in action
                rl4sys::cppclient::RL4SysAction rewarded_action = action; // copy to modify
                rewarded_action.updateReward(step_res.reward);
                agent.addToTrajectory(*traj, rewarded_action);

                obs = step_res.next_obs;
                cumulative_reward += step_res.reward;
                done = step_res.done;
                ++moves;
            }

            if (traj.has_value()) {
                // Create a dummy last action to satisfy interface if needed
                // (trajectory already has last action with reward, but we pass same ref)
                rl4sys::cppclient::RL4SysAction dummy_last_action;
                agent.markEndOfTrajectory(*traj, dummy_last_action);
            }

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();

            std::cout << "Iteration completed. Moves: " << moves
                      << ", Cumulative reward: " << cumulative_reward
                      << ", Time(ms): " << elapsed << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
