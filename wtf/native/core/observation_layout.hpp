#pragma once
#include "random.hpp"
namespace wtfcore {
inline const std::vector<std::string> &canonical_actions() {
    static const std::vector<std::string> names = {
        "move_forward", "move_backward", "strafe_left", "strafe_right",
        "turn_left",    "turn_right",    "rest",        "eat",
        "attack",       "swim",          "reproduce",   "challenge"};
    return names;
}
inline int action_index(const std::string &name) {
    const auto &a = canonical_actions();
    auto i = std::find(a.begin(), a.end(), name);
    if (i == a.end())
        throw std::runtime_error("Unknown canonical action");
    return static_cast<int>(i - a.begin());
}
inline int action_group(const std::string &name) {
    int i = action_index(name);
    return i < 4                 ? 0
           : i == 9              ? 1
           : i == 7              ? 2
           : (i == 8 || i == 11) ? 3
           : i == 10             ? 4
                                 : 5;
}
struct ObservationField {
    std::size_t raw = 0, count = 1, offset = 0, width = 1;
    std::string kind, label;
    std::vector<double> knots;
};
struct ObservationLayout {
    int version = 1, k = 1, history = 0;
    std::vector<std::string> raw_names, names, actions;
    std::vector<std::size_t> core_indices, confidence_offsets;
    std::vector<ObservationField> fields;
    std::vector<double> zeroes;
    std::size_t raw_core = 20, tracking = 0, physical_width = 20,
                consequence_start = 20, social_start = 20, scent_start = 20;
    explicit ObservationLayout(const Value &config);
    std::size_t width() const { return names.size(); }
    std::size_t offset(std::size_t raw) const;
    std::size_t index(const std::string &name) const;
    std::vector<double> encode_soft(const std::vector<double> &values) const;
    std::vector<double> decode_soft(const std::vector<double> &values,
                                    std::size_t raw_start = 0) const;
};
std::shared_ptr<const ObservationLayout>
shared_observation_layout(const Value &config);
std::vector<double> soft_memberships(double value,
                                     const std::vector<double> &knots);
std::vector<double> soft_direction(double forward, double right);
} // namespace wtfcore
