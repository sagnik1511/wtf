#pragma once
#include "random.hpp"
namespace wtfcore {
class World;
std::vector<std::uint32_t> offspring_seed_words(std::uint64_t world_seed,
                                                int id);
Value inherit_controller_state(const Value &mother, const Value &father,
                               const std::vector<std::uint32_t> &seed,
                               bool crossover = true);
void mutate_birth_state(World &world, Value &child_state, const Value &birth,
                        const std::vector<std::uint32_t> &seed);
void reset_birth_state(Value &state);
std::vector<Value *> mutation_parameter_arrays(Value &state);
} // namespace wtfcore
