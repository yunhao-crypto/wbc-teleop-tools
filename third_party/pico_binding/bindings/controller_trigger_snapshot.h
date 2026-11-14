#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <nlohmann/json.hpp>

// Both trigger (or grip) depths and the root timestamp from one controller packet.
struct ControllerTriggerSnapshot {
    bool available = false;
    int64_t packet_timestamp_ns = 0;
    double left = 0.0;
    double right = 0.0;
    std::chrono::steady_clock::time_point received{};
};

inline ControllerTriggerSnapshot parseControllerTriggerSnapshot(const nlohmann::json& value,
                                                                const char* key = "trigger") {
    ControllerTriggerSnapshot candidate;
    const auto& stamp = value.at("timeStampNs");
    if (!stamp.is_number_integer() || stamp.get<int64_t>() <= 0)
        throw std::runtime_error("Controller packet requires a positive root timestamp");
    const auto& controller = value.at("Controller");
    const auto& left = controller.at("left").at(key);
    const auto& right = controller.at("right").at(key);
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

// Forward (left Y) and turning (right X), from one packet with one receipt clock.
inline ControllerTriggerSnapshot parseControllerStickSnapshot(const nlohmann::json& value) {
    ControllerTriggerSnapshot out;
    const auto& stamp = value.at("timeStampNs");
    if (!stamp.is_number_integer() || stamp.get<int64_t>() <= 0)
        throw std::runtime_error("Invalid stick timestamp");
    const auto& l = value.at("Controller").at("left").at("axisY");
    const auto& r = value.at("Controller").at("right").at("axisX");
    if (!l.is_number() || !r.is_number()) throw std::runtime_error("Invalid stick axes");
    out.left = l.get<double>(); out.right = r.get<double>();
    if (!std::isfinite(out.left) || !std::isfinite(out.right) ||
        std::abs(out.left) > 1 || std::abs(out.right) > 1)
        throw std::runtime_error("Stick outside [-1,1]");
    out.packet_timestamp_ns = stamp.get<int64_t>(); out.available = true;
    return out;
}

struct ControllerTeleopSnapshot : ControllerTriggerSnapshot {
    bool left_click = false;
    bool right_click = false;
};

inline ControllerTeleopSnapshot parseControllerTeleopSnapshot(const nlohmann::json& value) {
    ControllerTeleopSnapshot out;
    const auto& stamp = value.at("timeStampNs");
    const auto& right = value.at("Controller").at("right");
    const auto& lc = value.at("Controller").at("left").at("axisClick");
    const auto& rc = right.at("axisClick");
    if (!stamp.is_number_integer() || stamp.get<int64_t>() <= 0 || !lc.is_boolean() || !rc.is_boolean()
            || !right.at("axisY").is_number() || !right.at("axisX").is_number())
        throw std::runtime_error("Invalid teleop controller packet");
    out.left=right.at("axisY").get<double>(); out.right=right.at("axisX").get<double>();
    if (!std::isfinite(out.left) || !std::isfinite(out.right) || std::abs(out.left)>1 || std::abs(out.right)>1)
        throw std::runtime_error("Invalid right-stick axes");
    out.left_click=lc.get<bool>(); out.right_click=rc.get<bool>();
    out.packet_timestamp_ns=stamp.get<int64_t>(); out.available=true;
    return out;
}
