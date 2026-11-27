#pragma once
#include "value.hpp"
namespace wtfcore {
Value native_model_signature(const Value &config,
                             const std::vector<std::string> &actions);
Value native_model_snapshot(const Value &controller, const Value &signature);
void load_native_model_weights(Value &controller, const Value &model,
                               const Value &signature);
// Initial metadata: config (validated), mode, initial_models, and optional
// randomization_seed/freeze_evaluation_brains. For a next episode, supply the
// finished experiment checkpoint with run_index incremented by the caller.
// No Python world/controller is constructed, restored or queried.
Value start_native_run(Value metadata);
} // namespace wtfcore
