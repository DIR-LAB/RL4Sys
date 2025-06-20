#include "utils/compression.h"
#include <zlib.h>
#include <sstream>

namespace rl4sys {
namespace utils {

std::vector<uint8_t> zlibDecompress(const std::vector<uint8_t>& compressed) {
    if (compressed.empty()) {
        return {};
    }

    z_stream strm = {};
    if (inflateInit(&strm) != Z_OK) {
        throw std::runtime_error("Failed to initialize zlib decompression");
    }

    strm.avail_in = static_cast<uInt>(compressed.size());
    strm.next_in = const_cast<Bytef*>(compressed.data());

    std::vector<uint8_t> decompressed;
    decompressed.reserve(compressed.size() * 4); // Initial estimate

    uint8_t buffer[32768];
    int ret;
    do {
        strm.avail_out = sizeof(buffer);
        strm.next_out = buffer;

        ret = inflate(&strm, Z_NO_FLUSH);
        switch (ret) {
            case Z_NEED_DICT:
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                inflateEnd(&strm);
                throw std::runtime_error("zlib decompression error: " + std::to_string(ret));
        }

        size_t have = sizeof(buffer) - strm.avail_out;
        decompressed.insert(decompressed.end(), buffer, buffer + have);
    } while (ret != Z_STREAM_END);

    inflateEnd(&strm);
    
    if (ret != Z_STREAM_END) {
        throw std::runtime_error("zlib decompression incomplete");
    }

    return decompressed;
}

std::vector<uint8_t> zlibCompress(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        return {};
    }

    z_stream strm = {};
    if (deflateInit(&strm, Z_DEFAULT_COMPRESSION) != Z_OK) {
        throw std::runtime_error("Failed to initialize zlib compression");
    }

    strm.avail_in = static_cast<uInt>(data.size());
    strm.next_in = const_cast<Bytef*>(data.data());

    std::vector<uint8_t> compressed;
    compressed.reserve(data.size()); // Initial estimate

    uint8_t buffer[32768];
    int ret;
    do {
        strm.avail_out = sizeof(buffer);
        strm.next_out = buffer;

        ret = deflate(&strm, Z_FINISH);
        if (ret == Z_STREAM_ERROR) {
            deflateEnd(&strm);
            throw std::runtime_error("zlib compression error");
        }

        size_t have = sizeof(buffer) - strm.avail_out;
        compressed.insert(compressed.end(), buffer, buffer + have);
    } while (ret != Z_STREAM_END);

    deflateEnd(&strm);
    return compressed;
}

} // namespace utils
} // namespace rl4sys 