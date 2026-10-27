#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <nlohmann/json.hpp>

// Both trigger depths and the root timestamp from one controller packet.
struct ControllerTriggerSnapshot {
    bool available = false;
    int64_t packet_timestamp_ns = 0;
    double left = 0.0;
    double right = 0.0;
    std::chrono::steady_clock::time_point received{};
};

inline ControllerTriggerSnapshot parseControllerTriggerSnapshot(const nlohmann::json& value) {
    ControllerTriggerSnapshot candidate;
    const auto& stamp = value.at("timeStampNs");
    if (!stamp.is_number_integer() || stamp.get<int64_t>() <= 0)
        throw std::runtime_error("Controller packet requires a positive root timestamp");
    const auto& controller = value.at("Controller");
    const auto& left = controller.at("left").at("trigger");
    const auto& right = controller.at("right").at("trigger");
    if (!left.is_number() || !right.is_number())
        throw std::runtime_error("Controller trigger depths must be numeric");
    candidate.left = left.get<double>();
    candidate.right = right.get<double>();
    if (!std::isfinite(candidate.left) || !std::isfinite(candidate.right)
            || candidate.left < 0 || candidate.left > 1 || candidate.right < 0 || candidate.right > 1)
        throw std::runtime_error("Controller trigger depth is outside [0,1]");
    candidate.packet_timestamp_ns = stamp.get<int64_t>();
    candidate.available = true;
    return candidate;
}
