#include "gtest/gtest.h"
#include "rl4sys_agent.h" // Include the header for the class we are testing
#include "config_loader.h"

#include <fstream> // To check if test config file exists
#include <stdexcept> // For exception checks

// Define the path to the test configuration relative to where the test runs
// This might need adjustment based on your build system's working directory for tests
const std::string TEST_CONFIG_PATH = "./test_conf.json";

namespace rl4sys {
namespace cppclient {
namespace test {

// Test fixture for RL4SysAgent tests (optional, but good practice)
class RL4SysAgentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure the test config file exists before running tests
        std::ifstream configFile(TEST_CONFIG_PATH);
        ASSERT_TRUE(configFile.is_open()) << "Test configuration file not found at: " << TEST_CONFIG_PATH;
    }

    // void TearDown() override {} // Cleanup if needed
};

// Test case to verify successful agent construction with a valid config file
TEST_F(RL4SysAgentTest, ConstructAgentSuccessfully) {
    // Expect that constructing the agent with a valid config file does not throw
    EXPECT_NO_THROW({
        try {
            RL4SysAgent agent(TEST_CONFIG_PATH);
            // We could add more checks here, e.g., verify config values loaded correctly
            // if we add public accessors to the AgentConfig within RL4SysAgent or
            // test ConfigLoader directly.
        } catch (const std::runtime_error& e) {
            // Provide more context if it fails during NO_THROW check
            FAIL() << "Agent construction threw an unexpected std::runtime_error: " << e.what();
        } catch (...) {
            FAIL() << "Agent construction threw an unexpected exception type.";
        }
    });
}

// Test case to verify agent construction fails with a non-existent config file
TEST_F(RL4SysAgentTest, ConstructAgentFailsWithMissingConfig) {
    std::string nonExistentPath = "./non_existent_config.json";
    // Expect that constructing the agent with an invalid path throws std::runtime_error
    // (specifically from the ConfigLoader trying to open the file)
    EXPECT_THROW({
        try {
            RL4SysAgent agent(nonExistentPath);
        } catch (const std::runtime_error& e) {
            // Optional: Check the error message contains expected text
            // EXPECT_THAT(e.what(), ::testing::HasSubstr("Could not open config file"));
            throw; // Re-throw to satisfy EXPECT_THROW
        }
    }, std::runtime_error);
}

// Add more tests here:
// - Test ConfigLoader directly to verify all fields are loaded correctly.
// - Test RL4SysAgent methods (requestForAction, addToTrajectory, markEndOfTrajectory)
//   (These would likely require a mock gRPC server)

} // namespace test
} // namespace cppclient
} // namespace rl4sys 