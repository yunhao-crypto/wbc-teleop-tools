#pragma once
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

// A complete body and the ROOT timestamp from the SAME device-state packet.
// Publication/copy is protected by the binding's bodyMutex.
struct BodyPoseSnapshot {
    bool available = false;
    int64_t packet_timestamp_ns = 0;
    std::array<std::array<double, 7>, 24> pose{};
};

inline BodyPoseSnapshot parseBodyPoseSnapshot(const nlohmann::json& value) {
    BodyPoseSnapshot candidate;
    const auto& stamp = value.at("timeStampNs");
    if (!stamp.is_number_integer()) throw std::runtime_error("Body packet timestamp must be an integer");
    candidate.packet_timestamp_ns = stamp.get<int64_t>();
    if (candidate.packet_timestamp_ns <= 0) throw std::runtime_error("Body packet timestamp must be positive");
    const auto& joints = value.at("Body").at("joints");
    if (!joints.is_array() || joints.size() != 24) throw std::runtime_error("Body packet must contain exactly 24 joints");
    for (size_t joint = 0; joint < 24; ++joint) {
        const std::string encoded = joints.at(joint).at("p").get<std::string>();
        std::stringstream stream(encoded);
        std::string token;
        size_t coordinate = 0;
        while (std::getline(stream, token, ',')) {
            if (coordinate >= 7) throw std::runtime_error("Body pose has more than 7 values");
            size_t used = 0;
            const double number = std::stod(token, &used);
            while (used < token.size() && std::isspace(static_cast<unsigned char>(token[used]))) ++used;
            if (used != token.size() || !std::isfinite(number)) throw std::runtime_error("Body pose has invalid or nonfinite value");
            candidate.pose[joint][coordinate++] = number;
        }
        if (coordinate != 7 || encoded.empty() || encoded.back() == ',') throw std::runtime_error("Body pose must contain exactly 7 values");
    }
    candidate.available = true;
    return candidate;
}
