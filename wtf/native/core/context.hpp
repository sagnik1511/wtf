#pragma once
#include "world.hpp"
#include <array>
namespace wtfcore {
// Valid for a single unchanged decision phase. Never advances clocks or RNG.
class DecisionContext {
    World &world_;
    std::map<CellPosition, int> positions_;
    std::set<CellPosition> occupied_;
    std::map<std::string, int> populations_;
    std::map<int, bool> eligible_;
    std::map<CellPosition, double> food_;

  public:
    explicit DecisionContext(World &world);
    bool eligible(const Value &animal) const;
    bool reproduction_available(const Value &animal) const;
    bool birth_space(const Value &first, const Value &second) const;
    std::vector<bool>
    action_mask(const Value &animal,
                const std::vector<std::string> &actions) const;
};
std::map<std::string, std::vector<double>> central_contexts(World &world);
} // namespace wtfcore
