#pragma once
#include "champions.hpp"
#include "controller.hpp"
#include "world.hpp"
#include <array>
#include <fstream>
#include <memory>
namespace wtfcore {
class Session {
    bool profile_ = false;
    std::array<double, 8> phase_seconds_{};
    Value state_;
    World world_;
    Champions champions_;
    std::map<int, std::unique_ptr<Controller>> brains_;
    std::map<int, Value> signatures_;
    std::map<int, std::pair<std::int64_t, Value>> model_cache_;
    Value model_state(int id);
    std::map<int, std::vector<double>> observations_;
    Value last_record_;
    bool finished_ = false;
    std::int64_t retired_ = 0;
    Value inherited_;
    std::int64_t capture_index_ = 1;
    int last_capture_ = 0, last_diagnostic_ = 0;
    void step();
    void finish_early_stopping();
    void refresh_context(
        const std::map<std::string, std::vector<double>> *contexts = nullptr);

  public:
    explicit Session(const Value &payload);
    Value run(int steps, const std::string &capture_path, const Value &options);
    Value checkpoint();
    Value observations() const;
    Value snapshot(bool diagnostics = false, bool activations = false);
    std::int64_t updates() const;
};
void write_json(std::ostream &stream, const Value &value);
} // namespace wtfcore
