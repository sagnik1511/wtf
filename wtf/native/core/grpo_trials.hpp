#pragma once
#include "controller.hpp"
#include "world.hpp"
namespace wtfcore {
// Returns exact focal PolicyStep records. Trial policies borrow live immutable
// parameters; only newborns own cloned/crossed/mutated controller state.
Value collect_group(World &world,
                    const std::map<int, std::unique_ptr<Controller>> &brains,
                    int focal_id);
} // namespace wtfcore
