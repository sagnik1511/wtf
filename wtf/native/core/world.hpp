#pragma once
#include "random.hpp"
#include <map>
#include <set>
#include <unordered_map>
namespace wtfcore {
class SocialState;
struct ObservationLayout;
using CellPosition = std::pair<int, int>;
struct TerrainGrid {
    int width = 0, height = 0;
    Value definition;
    std::vector<double> elevation, depth, flow_x, flow_y;
    std::vector<unsigned char> numeric_kinds;
    TerrainGrid(const Value &definition, int width, int height);
};
struct CompetitionResult {
    std::vector<int> priority;
    std::set<int> deaths, successful;
};
struct WorldStep {
    std::map<int, double> rewards;
    std::vector<int> dead;
    Value events = Value::array();
};
// Ordered native state preserves source iteration and tie order. No Python
// owner.
class World {
    std::unique_ptr<SocialState> social_;
    std::map<std::string, std::shared_ptr<const ObservationLayout>> layouts_;
    int width_, height_;
    std::vector<unsigned char> blocked_;
    std::unordered_map<int, std::size_t> animal_index_;
    void index_animals();
    std::shared_ptr<const TerrainGrid> terrain_;
    mutable int body_tick_ = -1;
    mutable Value cycle_bodies_ = Value::object();
    void refresh_bodies() const;
    void river_drift(Value &events);
    CompetitionResult
    resolve_competition(const std::map<int, std::string> &actions,
                        Value &events);
    void contest_advance(int winner, int loser, Value &events,
                         std::set<CellPosition> &occupied);

  public:
    Value state;
    Value config;
    Value frame;
    Random rng;
    Value last_dead_animals = Value::array();
    explicit World(const Value &checkpoint,
                   std::shared_ptr<const TerrainGrid> static_geography = {});
    ~World();
    SocialState *social() { return social_.get(); }
    std::shared_ptr<const ObservationLayout>
    observation_layout(const std::string &name);
    int width() const { return width_; }
    int height() const { return height_; }
    int tick() const { return frame.integer("tick"); }
    Value::Array &animals() { return frame["animals"].items(); }
    const Value::Array &animals() const { return frame.at("animals").items(); }
    Value::Array &resources() { return frame["resources"].items(); }
    const Value::Array &resources() const {
        return frame.at("resources").items();
    }
    Value &animal(int id);
    const Value &animal(int id) const;
    const Value &species(const Value &animal) const {
        return config.at("species").at(animal.text("species"));
    }
    const Value &body(const Value &animal) const;
    const Value &body_for_species(const std::string &name) const;
    bool has_terrain() const { return bool(terrain_); }
    std::shared_ptr<const TerrainGrid> geography() const { return terrain_; }
    double light() const;
    Value elevation_value(int x, int y) const;
    Value depth_value(int x, int y) const;
    std::pair<Value, Value> flow_values(int x, int y) const;
    double elevation(int x, int y) const {
        return elevation_value(x, y).number();
    }
    double depth(int x, int y) const { return depth_value(x, y).number(); }
    std::pair<double, double> flow(int x, int y) const;
    CellPosition movement_destination(const Value &animal,
                                      const std::string &action) const;
    Value terrain_move_cost(const Value &animal, int x, int y,
                            const std::string &action) const;
    std::string movement_block_reason(
        const Value &animal, int x, int y, const std::string &action,
        const std::set<CellPosition> *occupied = nullptr) const;
    std::string movement_reason(const Value &animal, int x, int y,
                                const std::string &action, bool occupied) const;
    void terrain_charge(Value &animal, const Value &amount, Value &events,
                        const std::string &cause);
    int water_sight_cells(int x0, int y0, int x1, int y1) const;
    bool mating_land_route(const Value &first, const Value &second,
                           int radius) const;
    bool birth_cell_allowed(const Value &parent, int x, int y) const;
    std::pair<int, std::string>
    challenge_opportunity(const Value &animal,
                          const std::map<CellPosition, int> &positions) const;
    bool blocked(int x, int y) const;
    bool visible(int x0, int y0, int x1, int y1) const;
    WorldStep advance(const std::map<int, std::string> &actions);
    Value checkpoint();
};
std::map<int, std::vector<double>> observe(World &world);
Value world_metrics(World &world);
} // namespace wtfcore
