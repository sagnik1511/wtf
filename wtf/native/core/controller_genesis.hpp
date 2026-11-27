#pragma once
#include "observation_layout.hpp"
namespace wtfcore {
// Resolved templates and unsigned little-endian seed words; no Python objects
// or shared mutable policy state enter controller construction.
Value make_controller_state(const Value &config,
                            const std::vector<std::string> &actions,
                            const std::vector<std::uint32_t> &base_seed,
                            const std::vector<std::uint32_t> &individual_seed);
} // namespace wtfcore
