#include "session.hpp"
#include "context.hpp"
#include "grpo_trials.hpp"
#include "inheritance.hpp"
#include "social.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
namespace wtfcore {
namespace {
void escaped(std::ostream &stream, const std::string &text) {
    stream << '"';
    const char *hex = "0123456789abcdef";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\')
            stream << '\\' << c;
        else if (c < 32)
            stream << "\\u00" << hex[c >> 4] << hex[c & 15];
        else
            stream << c;
    }
    stream << '"';
}
} // namespace
void write_json(std::ostream &stream, const Value &v) {
    if (v.null()) {
        stream << "null";
        return;
    }
    if (auto p = std::get_if<bool>(&v.data)) {
        stream << (*p ? "true" : "false");
        return;
    }
    if (auto p = std::get_if<std::int64_t>(&v.data)) {
        stream << *p;
        return;
    }
    if (auto p = std::get_if<std::uint64_t>(&v.data)) {
        stream << *p;
        return;
    }
    if (auto p = std::get_if<double>(&v.data)) {
        if (!std::isfinite(*p))
            throw std::runtime_error("Nonfinite native artifact value");
        char buffer[64];
        auto result = std::to_chars(buffer, buffer + sizeof(buffer), *p,
                                    std::chars_format::general);
        if (result.ec != std::errc())
            throw std::runtime_error("Native floating encoding failed");
        std::string s(buffer, result.ptr);
        if (s.find_first_of(".eE") == std::string::npos)
            s += ".0";
        stream << s;
        return;
    }
    if (auto p = std::get_if<std::string>(&v.data)) {
        escaped(stream, *p);
        return;
    }
    if (auto holder = std::get_if<Value::FloatArrayPtr>(&v.data)) {
        stream << '[';
        bool first = true;
        for (float number : **holder) {
            if (!first)
                stream << ',';
            first = false;
            write_json(stream, Value(static_cast<double>(number)));
        }
        stream << ']';
        return;
    }
    if (auto holder = std::get_if<Value::ArrayPtr>(&v.data)) {
        const auto *p = holder->get();
        stream << '[';
        bool first = true;
        for (const auto &item : *p) {
            if (!first)
                stream << ',';
            first = false;
            write_json(stream, item);
        }
        stream << ']';
        return;
    }
    stream << '{';
    bool first = true;
    for (const auto &[key, value] : v.fields()) {
        if (!first)
            stream << ',';
        first = false;
        escaped(stream, key);
        stream << ':';
        write_json(stream, value);
    }
    stream << '}';
}
Session::Session(const Value &payload)
    : state_(payload.at("checkpoint")), world_(state_.at("world")),
      champions_(state_.at("config"), state_.at("champions"),
                 state_.at("lineage")),
      retired_(state_.at("retired_updates").i64()),
      inherited_(state_.at("inherited_updates")) {
    const auto &mode = state_.at("mode").str();
    if (mode != "train" && mode != "evaluate" && mode != "random")
        throw std::runtime_error(
            "Native execution requires train, evaluate or random mode");
    if (retired_ < 0)
        throw std::runtime_error(
            "Native retired update count must be nonnegative");
    finished_ = state_.flag("finished_run");
    for (const auto &[id, brain] : state_.at("brains").fields()) {
        brains_[std::stoi(id)] = restore_controller(brain);
        brains_[std::stoi(id)]->use_world_inputs();
    }
    for (const auto &[id, signature] : payload.at("signatures").fields())
        signatures_[std::stoi(id)] = signature;
    for (const auto &[id, obs] : payload.at("observations").fields())
        observations_[std::stoi(id)] = numbers(obs);
    if (world_.social())
        world_.social()->set_training(mode == "train");
    refresh_context();
    // The imported boundary tree is not a second live model/rollout bank.
    for (const auto *key : {"world", "brains", "champions", "lineage"})
        state_[key] = Value{};
    last_capture_ = world_.tick();
    last_diagnostic_ = world_.tick();
    (void)updates();
}
void Session::refresh_context(
    const std::map<std::string, std::vector<double>> *contexts) {
    bool masks = false, central = false;
    for (const auto &a : world_.animals()) {
        const auto &kind = brains_.at(a.integer("id"))->kind();
        masks |= kind == "maskable_ppo";
        central |= kind == "mappo";
    }
    std::unique_ptr<DecisionContext> decision;
    if (masks)
        decision = std::make_unique<DecisionContext>(world_);
    std::map<std::string, std::vector<double>> fresh;
    if (central && !contexts) {
        fresh = central_contexts(world_);
        contexts = &fresh;
    }
    for (const auto &a : world_.animals()) {
        auto &brain = *brains_.at(a.integer("id"));
        if (brain.kind() == "maskable_ppo") {
            Value mask = Value::array();
            for (bool valid : decision->action_mask(a, brain.actions()))
                mask.push(valid);
            brain.set_context(object({{"mask", std::move(mask)}}));
        } else if (brain.kind() == "mappo") {
            auto current = values(contexts->at(a.text("species")));
            brain.set_context(
                object({{"context", current}, {"next_context", current}}));
        }
    }
}
Value Session::model_state(int id) {
    auto &brain = *brains_.at(id);
    auto found = model_cache_.find(id);
    // Every live parameter mutation belongs to this private native controller;
    // callers cannot modify arrays or install Python hooks during execution.
    if (found == model_cache_.end() || found->second.first != brain.updates())
        model_cache_[id] = {brain.updates(), brain.model_state()};
    return model_cache_.at(id).second;
}
std::int64_t Session::updates() const {
    std::int64_t n = retired_;
    for (const auto &[id, brain] : brains_) {
        const auto &saved = inherited_.get(std::to_string(id));
        std::int64_t inherited = saved.null() ? 0 : saved.i64();
        if (inherited < 0 || inherited > brain->updates())
            throw std::runtime_error("Invalid native inherited update count");
        std::int64_t delta = brain->updates() - inherited;
        if (n > std::numeric_limits<std::int64_t>::max() - delta)
            throw std::runtime_error(
                "Native total update counter capacity exceeded");
        n += delta;
    }
    return n;
}
void Session::step() {
    if (finished_)
        return;
    using Clock = std::chrono::steady_clock;
    auto stamp = profile_ ? Clock::now() : Clock::time_point{};
    auto lap = [&](std::size_t phase) {
        if (profile_) {
            auto now = Clock::now();
            phase_seconds_[phase] +=
                std::chrono::duration<double>(now - stamp).count();
            stamp = now;
        }
    };
    bool training = state_.text("mode") == "train";
    if (training) {
        std::vector<int> grouped;
        for (const auto &a : world_.animals())
            if (brains_.at(a.integer("id"))->kind() == "grpo")
                grouped.push_back(a.integer("id"));
        if (!grouped.empty()) {
            std::size_t start = world_.tick() % grouped.size();
            int budget = state_.at("config")
                             .at("experiment")
                             .integer("grpo_updates_per_step", 1);
            for (std::size_t offset = 0; offset < grouped.size(); ++offset) {
                int id = grouped[(start + offset) % grouped.size()];
                if (brains_.at(id)->needs_group()) {
                    brains_.at(id)->update_group(
                        collect_group(world_, brains_, id));
                    if (!--budget)
                        break;
                }
            }
        }
    }
    std::map<int, std::string> previous_species;
    std::vector<int> previous_ids;
    previous_ids.reserve(world_.animals().size());
    for (const auto &animal : world_.animals())
        previous_ids.push_back(animal.integer("id"));
    bool central = false;
    if (training)
        for (const auto &a : world_.animals()) {
            int id = a.integer("id");
            previous_species[id] = a.text("species");
            central |= brains_.at(id)->kind() == "mappo";
        }
    std::map<int, std::string> actions;
    for (const auto &[id, obs] : observations_)
        actions[id] = brains_.at(id)->choose(obs, training,
                                             state_.text("mode") == "random");
    lap(0);
    auto result = world_.advance(actions);
    lap(1);
    auto next = observe(world_);
    lap(2);
    bool limit =
        world_.tick() >= world_.config.at("world").integer("max_steps");
    std::set<int> dead(result.dead.begin(), result.dead.end());
    std::map<std::string, std::vector<double>> contexts;
    std::map<std::string, std::pair<Sum, std::size_t>> team_rewards;
    if (central) {
        contexts = central_contexts(world_);
        for (int id : previous_ids) {
            auto &team = team_rewards[previous_species.at(id)];
            team.first.add(result.rewards.at(id));
            ++team.second;
        }
        for (const auto &[id, obs] : observations_)
            if (brains_.at(id)->kind() == "mappo")
                brains_.at(id)->set_context(
                    object({{"next_context",
                             values(contexts.at(previous_species.at(id)))}}));
    }
    if (training)
        for (const auto &[id, obs] : observations_) {
            auto found = next.find(id);
            double reward = result.rewards.at(id);
            if (brains_.at(id)->kind() == "mappo") {
                auto team = team_rewards.at(previous_species.at(id));
                reward = team.first.result() / team.second;
            }
            brains_.at(id)->learn(obs, actions.at(id), reward,
                                  found == next.end() ? nullptr
                                                      : &found->second,
                                  dead.count(id), limit && !dead.count(id));
        }
    lap(3);
    Value::Array tracked, newborns;
    for (int id : previous_ids) {
        if (dead.count(id)) {
            for (const auto &a : world_.last_dead_animals.items())
                if (a.integer("id") == id)
                    tracked.push_back(a);
        } else
            tracked.push_back(world_.animal(id));
    }
    for (const auto &event : result.events.items())
        if (event.text("kind") == "born")
            newborns.push_back(world_.animal(event.integer("target")));
    if (training) {
        auto source = object({{"experiment", state_.at("config").at("name")},
                              {"mode", state_.at("mode")},
                              {"run", state_.integer("run_index") + 1},
                              {"seed", world_.state.at("seed")},
                              {"tick", world_.tick()}});
        champions_.consider(
            tracked, [&](int id) { return model_state(id); }, source, newborns,
            signatures_);
    }
    lap(4);
    for (const auto &event : result.events.items())
        if (event.text("kind") == "born") {
            int id = event.integer("target"), mother = event.integer("actor"),
                father = event.integer("mate");
            auto seed = offspring_seed_words(world_.state.at("seed").u64(), id);
            if (!equivalent(signatures_.at(mother), signatures_.at(father)))
                throw std::runtime_error(
                    "Incompatible native parent contracts");
            Value child;
            if (state_.text("mode") == "evaluate" &&
                state_.flag("freeze_evaluation_brains")) {
                int parent =
                    world_.animal(id).text("sex") == "female" ? mother : father;
                child = inherit_controller_state(
                    brains_.at(parent)->checkpoint(), Value{}, seed, false);
            } else
                child = inherit_controller_state(
                    brains_.at(mother)->checkpoint(),
                    brains_.at(father)->checkpoint(), seed);
            if (training)
                mutate_birth_state(world_, child, event, seed);
            brains_[id] = restore_controller(child);
            brains_[id]->use_world_inputs();
            inherited_[std::to_string(id)] = brains_.at(id)->updates();
            signatures_[id] = signatures_.at(mother);
        }
    for (int id : result.dead) {
        if (state_.flag("legacy_brain_retention")) {
            int founders = 0;
            for (const auto &[name, spec] :
                 world_.config.at("species").fields())
                founders += spec.integer("initial_population");
            if (id < founders)
                continue;
        }
        const auto &saved = inherited_.get(std::to_string(id));
        std::int64_t inherited = saved.null() ? 0 : saved.i64();
        if (inherited < 0 || inherited > brains_.at(id)->updates())
            throw std::runtime_error("Invalid native inherited update count");
        std::int64_t delta = brains_.at(id)->updates() - inherited;
        if (retired_ > std::numeric_limits<std::int64_t>::max() - delta)
            throw std::runtime_error(
                "Native retired update counter capacity exceeded");
        retired_ += delta;
        brains_.erase(id);
        model_cache_.erase(id);
        signatures_.erase(id);
        inherited_.erase(std::to_string(id));
    }
    lap(5);
    observations_ = std::move(next);
    refresh_context(central ? &contexts : nullptr);
    finished_ = limit || world_.animals().empty();
    last_record_ = Value{};
    if (world_.tick() % state_.at("config")
                            .at("experiment")
                            .integer("record_metrics_every_steps") ==
            0 ||
        finished_) {
        last_record_ = world_metrics(world_);
        auto &history = state_["history"].items();
        history.push_back(last_record_);
        if (history.size() > 2000)
            history.erase(history.begin(), history.end() - 2000);
    }
    if (finished_) {
        Value summary = object({{"run", state_.integer("run_index") + 1},
                                {"seed", world_.state.at("seed")}}),
              initialized = Value::object(), survivors = Value::array();
        for (const auto &[species, groups] :
             state_.at("run_initial_models").fields())
            for (const auto &[sex, model] : groups.fields())
                initialized[species][sex] = model.at("provenance");
        summary["initialized_from"] = initialized;
        for (const auto &[key, value] : last_record_.fields())
            summary[key] = value;
        summary["lifespans"] = world_.state.at("lifespans");
        for (const auto &a : world_.animals())
            survivors.push(object({{"id", a.at("id")},
                                   {"species", a.at("species")},
                                   {"age", a.at("age")}}));
        summary["survivors"] = survivors;
        state_["summaries"].push(summary);
        finish_early_stopping();
    }
    lap(6);
}
void Session::finish_early_stopping() {
    auto &early = state_["early_stopping"];
    if (early.null())
        return;
    int run = state_.integer("run_index") + 1;
    int patience = early.integer("patience_runs");
    if (patience > 0 && state_.text("mode") == "train" &&
        !early.flag("stopped") && run > early.integer("last_run")) {
        bool improved = false;
        const auto &entries = champions_.entries();
        for (const auto &[species, groups] : early.at("criteria").fields())
            for (const auto &[sex, criterion] : groups.fields()) {
                const auto &entry =
                    entries.get(species).get(sex).get(criterion.str());
                if (entry.null())
                    continue;
                const auto &previous =
                    early.get("best_ranks").get(species).get(sex);
                bool better = previous.null() ||
                              previous.text("criterion") != criterion.str();
                if (!better) {
                    const auto &a = entry.at("rank").items();
                    const auto &b = previous.at("rank").items();
                    for (std::size_t i = 0; i < std::min(a.size(), b.size());
                         ++i) {
                        if (a[i].number() != b[i].number()) {
                            better = a[i].number() > b[i].number();
                            break;
                        }
                        if (i + 1 == std::min(a.size(), b.size()))
                            better = a.size() > b.size();
                    }
                }
                if (better) {
                    early["best_ranks"][species][sex] = object(
                        {{"criterion", criterion}, {"rank", entry.at("rank")}});
                    improved = true;
                }
            }
        early["stagnant_runs"] =
            improved ? 0 : early.integer("stagnant_runs") + 1;
        early["last_run"] = run;
        if (early.integer("stagnant_runs") >= patience) {
            early["stopped"] = true;
            early["reason"] =
                object({{"cause", "no_champion_improvement"},
                        {"run", run},
                        {"stagnant_runs", early.at("stagnant_runs")},
                        {"patience_runs", patience},
                        {"species", early.at("species")}});
        }
    }
    if (patience)
        state_["summaries"].items().back()["early_stopping"] = early;
}
Value Session::checkpoint() {
    Value out = state_;
    out["world"] = world_.checkpoint();
    out["brains"] = Value::object();
    for (auto &[id, brain] : brains_)
        out["brains"][std::to_string(id)] = brain->checkpoint();
    out["champions"] = champions_.entries();
    out["lineage"] = champions_.lineage();
    out["retired_updates"] = retired_;
    out["inherited_updates"] = inherited_;
    out["finished_run"] = finished_;
    return out;
}

Value Session::observations() const {
    Value out = Value::object();
    for (const auto &animal : world_.animals()) {
        int id = animal.integer("id");
        const auto &obs = observations_.at(id);
        Value row = world_observation_values(
            obs, brains_.at(id)->config().integer("observation_version", 1));
        out[std::to_string(id)] = std::move(row);
    }
    return out;
}
Value Session::snapshot(bool diagnostics, bool activations) {
    Value out = world_.frame;
    out["metrics"] = world_metrics(world_);
    out["total_updates"] = updates();
    out["finished_run"] = finished_;
    if (diagnostics) {
        out["diagnostics"] = Value::object();
        for (const auto &animal : world_.animals()) {
            int id = animal.integer("id");
            const auto &observation = observations_.at(id);
            out["diagnostics"][std::to_string(id)] =
                brains_.at(id)->diagnostics(observation, activations);
        }
    }
    return out;
}
Value Session::run(int steps, const std::string &path, const Value &options) {
    if (steps < -1)
        throw std::runtime_error("Steps must be -1 or nonnegative");
    profile_ = options.flag("profile_native");
    phase_seconds_.fill(0.0);
    int budget = steps < 0 ? std::numeric_limits<int>::max() : steps;
    std::ofstream stream;
    if (!path.empty()) {
        stream.open(path, std::ios::out | std::ios::app);
        if (!stream)
            throw std::runtime_error("Unable to open native capture spool");
    }
    int stride = options.integer("frame_every_steps", 1),
        diagnostic = options.integer("diagnostics_every_steps", 10),
        per_chunk = options.integer("frames_per_chunk", 64);
    if (stride < 1 || diagnostic < 0 || per_chunk < 1)
        throw std::runtime_error("Invalid native capture cadence");
    while (budget-- && !finished_) {
        step();
        auto capture_start = profile_ ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        if (!path.empty() && (finished_ || world_.tick() % stride == 0) &&
            world_.tick() != last_capture_) {
            bool sample =
                diagnostic && (capture_index_ % per_chunk == 0 || finished_ ||
                               world_.tick() - last_diagnostic_ >= diagnostic);
            auto frame = snapshot(sample, options.flag("record_activations"));
            for (const auto *key :
                 {"width", "height", "resource_capacity", "obstacles"})
                frame.erase(key);
            if (sample) {
                for (auto &[id, value] : frame["diagnostics"].fields()) {
                    Value entry = object({{"tick", world_.tick()}});
                    for (const auto &[key, item] : value.fields())
                        entry[key] = item;
                    value = std::move(entry);
                }
                last_diagnostic_ = world_.tick();
            }
            write_json(stream, frame);
            stream << '\n';
            if (!stream)
                throw std::runtime_error("Native capture write failed");
            last_capture_ = world_.tick();
            ++capture_index_;
        }
        if (profile_)
            phase_seconds_[7] +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              capture_start)
                    .count();
    }
    if (!path.empty()) {
        stream.flush();
        if (!stream)
            throw std::runtime_error("Native capture flush failed");
    }
    Value report =
        object({{"tick", world_.tick()},
                {"finished_run", finished_},
                {"early_stopped", state_.get("early_stopping").flag("stopped")},
                {"total_updates", updates()},
                {"last_record", last_record_}});
    if (profile_) {
        const char *names[] = {"choose",    "physics",   "observe", "learn",
                               "champions", "lifecycle", "metrics", "capture"};
        report["profile_native"] = Value::object();
        for (std::size_t i = 0; i < phase_seconds_.size(); ++i)
            report["profile_native"][names[i]] = phase_seconds_[i];
    }
    return report;
}
} // namespace wtfcore
