#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>

namespace rl4sys {
namespace utils {

/**
 * @brief Decompress data using zlib compression.
 * 
 * This handles the application-level zlib compression used by the Python server
 * for model state serialization. Note that gRPC transport compression is handled
 * automatically by the gRPC library.
 * 
 * @param compressed The zlib-compressed data
 * @return The decompressed data
 * @throws std::runtime_error if decompression fails
 */
std::vector<uint8_t> zlibDecompress(const std::vector<uint8_t>& compressed);

/**
 * @brief Compress data using zlib compression.
 * 
 * This is mainly used for testing and debugging purposes.
 * 
 * @param data The data to compress
 * @return The zlib-compressed data
 * @throws std::runtime_error if compression fails
 */
std::vector<uint8_t> zlibCompress(const std::vector<uint8_t>& data);

} // namespace utils
} // namespace rl4sys 