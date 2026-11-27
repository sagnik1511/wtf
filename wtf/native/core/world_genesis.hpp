#pragma once
#include "world.hpp"
namespace wtfcore {
// Validated, resolved configuration only. No Python construction or
// observation. The caller attaches initial SocialState before first observation
// when needed.
Value make_world(const Value &config, std::uint64_t seed);
} // namespace wtfcore
