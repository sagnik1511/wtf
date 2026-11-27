#pragma once
#include "value.hpp"
#include <functional>
#include <map>
namespace wtfcore {
// Detached species/sex champion records and compact founder-family ledger.
class Champions {
    Value config_, entries_, lineage_, policies_;
    void add_node(int id, const Value &node, const Value &previous,
                  bool is_new);
    void observe_lineage(const Value::Array &animals);
    std::pair<Value, Value> lineage_rank(const Value &animal,
                                         bool metadata) const;

  public:
    Champions(const Value &config, const Value &entries, const Value &lineage);
    void consider(const Value::Array &tracked_poststep,
                  const std::function<Value(int)> &brain_provider,
                  const Value &source, const Value::Array &newborns,
                  const std::map<int, Value> &signatures);
    void consider(const Value::Array &tracked_poststep,
                  const std::map<int, Value> &brain_checkpoints,
                  const Value &source, const Value::Array &newborns,
                  const std::map<int, Value> &signatures) {
        consider(
            tracked_poststep, [&](int id) { return brain_checkpoints.at(id); },
            source, newborns, signatures);
    }
    Value entries() const { return entries_; }
    Value lineage() const { return lineage_; }
};
} // namespace wtfcore
