#pragma once
#include "value.hpp"
#include <array>
#include <map>
#include <memory>
namespace wtfcore {
class World;
class ConsequenceModel {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    using Context = std::array<int, 12>;
    using Labels = std::array<std::int64_t, 3>;
    using Rates = std::array<double, 4>;
    explicit ConsequenceModel(const Value &config,
                              const Value &state = Value{});
    ~ConsequenceModel();
    ConsequenceModel(ConsequenceModel &&) noexcept;
    ConsequenceModel &operator=(ConsequenceModel &&) noexcept;
    bool has_species(const std::string &species) const;
    int horizon() const;
    void learn(const std::string &species, const Context &context,
               const Labels &labels);
    Rates predict(const std::string &species, const Context &context) const;
    Rates predict_action(const std::string &species, int action) const;
    Value checkpoint() const;
    Value export_state() const;
    void publish();
};
class SocialState {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit SocialState(const Value &config, const Value &state = Value{});
    ~SocialState();
    SocialState(SocialState &&) noexcept;
    SocialState &operator=(SocialState &&) noexcept;
    // Explicit observation-phase invalidation: call once before an observe-all
    // batch, and after any same-tick native edits outside the ordinary step.
    void prepare(World &world);
    void begin(World &world, const std::map<int, std::string> &actions);
    void finish(World &world, const Value &events, const Value &dead_animals);
    std::vector<double>
    observe(World &world, const Value &animal, const Value &brain_template,
            const std::vector<std::string> *compact_actions = nullptr);
    Value checkpoint() const;
    bool training() const;
    void set_training(bool training);
    bool has_knowledge() const;
    Value knowledge_checkpoint() const;
    Value knowledge_export() const;
    void publish();
};
} // namespace wtfcore
