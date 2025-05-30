#pragma once

#include <vector>
#include <string>
#include <optional>
#include <cstdint> // For fixed-width integers like int64_t
#include <map>

// Forward declare protobuf message types to avoid including proto headers here if possible
namespace rl4sys_proto {
    class Action;
    class Trajectory;
    class Observation;
} // namespace rl4sys_proto


namespace rl4sys {
namespace cppclient {

/**
 * @brief Represents a single action recommended by the RL agent.
 *
 * This class encapsulates the action itself and provides methods
 * to manage associated data like rewards.
 */
class RL4SysAction {
public:
    /**
     * @brief Default constructor.
     */
    RL4SysAction();

    /**
     * @brief Constructor with parameters.
     * @param obs Observation data.
     * @param act Action value.
     * @param reward Reward value.
     * @param done Whether this is the final action.
     * @param data Extra data map.
     * @param ver Version number.
     */
    RL4SysAction(const std::vector<double>& obs, int64_t act, double reward, bool done, 
                 const std::map<std::string, std::string>& data, int ver);

    /**
     * @brief Updates the reward associated with this action.
     * @param reward The reward value received after taking this action.
     */
    void updateReward(double reward);

    /**
     * @brief Gets the action value.
     * @return The action value (specific interpretation depends on the environment).
     *         Using int64_t as a placeholder; adjust based on actual action space.
     */
    int64_t getActionValue() const;

    /**
     * @brief Sets the action value.
     * @param value The action value.
     */
    void setActionValue(int64_t value);

    /**
     * @brief Gets the reward associated with this action, if set.
     * @return An optional containing the reward, or nullopt if not set.
     */
    std::optional<double> getReward() const;

    /**
     * @brief Gets the observation data.
     * @return The observation vector.
     */
    const std::vector<double>& getObservation() const;

    /**
     * @brief Checks if reward is set.
     * @return True if reward is set, false otherwise.
     */
    bool is_reward_set() const;

    /**
     * @brief Checks if this is the final action.
     * @return True if this is the final action, false otherwise.
     */
    bool is_done() const;

    /**
     * @brief Sets the done flag.
     * @param done Whether this is the final action.
     */
    void set_done(bool done);

    /**
     * @brief Gets the extra data map.
     * @return The extra data map.
     */
    const std::map<std::string, std::string>& getData() const;

    /**
     * @brief Gets the version number.
     * @return The version number.
     */
    int getVersion() const;

    // Add other necessary methods and members based on rl4sys.proto Action message
    // --- Added Example Members ---

private:
    // Internal representation, possibly mapping to rl4sys_proto::Action
    std::vector<double> observation;
    int64_t actionValue; // Example: Use appropriate type
    std::optional<double> actionReward;
    bool done;
    std::map<std::string, std::string> extraData;
    int version;
    // Other fields corresponding to the protobuf Action message
};


/**
 * @brief Represents a sequence of observations, actions, and rewards.
 *
 * This class manages the data collected during an episode or part of an episode,
 * preparing it for transmission to the training server.
 */
class RL4SysTrajectory {
public:
    /**
     * @brief Default constructor.
     */
    RL4SysTrajectory();

    /**
     * @brief Adds an action to the trajectory.
     * @param action The RL4SysAction object to add.
     */
    void addAction(const RL4SysAction& action);

    /**
     * @brief Adds an observation to the trajectory.
     * @param observation The observation data (adjust vector type as needed, e.g., float).
     */
    void addObservation(const std::vector<double>& observation);


    /**
     * @brief Checks if the trajectory is empty.
     * @return True if the trajectory contains no steps, false otherwise.
     */
    bool isEmpty() const;

    /**
     * @brief Clears the trajectory data.
     */
    void clear();


    // Add other necessary methods and members based on rl4sys.proto Trajectory message
    // --- Added Example Accessors ---
    /**
     * @brief Gets the observations stored in the trajectory.
     * @return A constant reference to the vector of observations.
     */
    const std::vector<std::vector<double>>& getObservations() const;

    /**
     * @brief Gets the actions stored in the trajectory.
     * @return A constant reference to the vector of actions.
     */
    const std::vector<RL4SysAction>& getActions() const;


private:
    // Internal representation, possibly mapping to rl4sys_proto::Trajectory
    std::vector<std::vector<double>> observations; // Example type
    std::vector<RL4SysAction> actions;
    // Other fields corresponding to the protobuf Trajectory message, like 'done' flag
    bool isDone = false; // Example: Flag to indicate if trajectory is terminal
};

} // namespace cppclient
} // namespace rl4sys
