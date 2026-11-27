#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "context.hpp"
#include "controller.hpp"
#include "controller_genesis.hpp"
#include "grpo_trials.hpp"
#include "inheritance.hpp"
#include "observation_layout.hpp"
#include "ppo.hpp"
#include "run_genesis.hpp"
#include "session.hpp"
#include "world.hpp"
#include "world_genesis.hpp"
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
namespace {
using namespace wtfcore;
struct RecursionGuard {
    RecursionGuard() {
        if (Py_EnterRecursiveCall(" while converting native engine data"))
            throw std::runtime_error("Native boundary nesting exceeded");
    }
    ~RecursionGuard() { Py_LeaveRecursiveCall(); }
};
// Own a single boundary reference, including when C++ allocation throws.
struct PyRef {
    PyObject *value;
    explicit PyRef(PyObject *object) : value(object) {}
    ~PyRef() { Py_XDECREF(value); }
    PyRef(const PyRef &) = delete;
    PyRef &operator=(const PyRef &) = delete;
    PyObject *release() {
        auto *out = value;
        value = nullptr;
        return out;
    }
};
// A borrowed buffer remains pinned and is released even when native allocation
// or memo insertion throws. Only an explicit native one-dimensional FP32 buffer
// enters packed storage; ordinary float lists retain their existing semantics.
struct BufferView {
    Py_buffer view{};
    ~BufferView() {
        if (view.obj)
            PyBuffer_Release(&view);
    }
};
// Boundary-scoped memos retain DAG sharing without retaining PyObjects in the
// native runtime. Keeping input references also prevents iterator-address
// reuse.
struct FromMemo {
    std::unordered_map<PyObject *, Value> values;
    ~FromMemo() {
        for (const auto &[object, value] : values)
            Py_DECREF(object);
    }
    void remember(PyObject *object, const Value &value) {
        auto found = values.emplace(object, value);
        if (found.second)
            Py_INCREF(object);
    }
};
Value from_python(PyObject *object, FromMemo &memo) {
    if (object == Py_None)
        return Value{};
    if (PyBool_Check(object))
        return object == Py_True;
    if (PyLong_Check(object)) {
        int overflow = 0;
        auto n = PyLong_AsLongLongAndOverflow(object, &overflow);
        if (!overflow && !PyErr_Occurred())
            return static_cast<std::int64_t>(n);
        PyErr_Clear();
        auto u = PyLong_AsUnsignedLongLong(object);
        if (PyErr_Occurred())
            throw std::runtime_error("Integer exceeds native 64-bit boundary");
        return static_cast<std::uint64_t>(u);
    }
    if (PyFloat_Check(object))
        return PyFloat_AS_DOUBLE(object);
    if (PyUnicode_Check(object)) {
        Py_ssize_t length = 0;
        const char *text = PyUnicode_AsUTF8AndSize(object, &length);
        if (!text)
            throw std::runtime_error("Invalid native text");
        return std::string(text, static_cast<std::size_t>(length));
    }
    auto cached = memo.values.find(object);
    if (cached != memo.values.end())
        return cached->second;
    RecursionGuard recursion;
    PyRef input(Py_NewRef(object));
    if (PyObject_CheckBuffer(object)) {
        BufferView buffer;
        if (PyObject_GetBuffer(object, &buffer.view,
                               PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0)
            throw std::runtime_error("Unable to read native boundary buffer");
        const auto &view = buffer.view;
        if (view.format && std::strcmp(view.format, "f") == 0 &&
            view.itemsize == static_cast<Py_ssize_t>(sizeof(float))) {
            if (view.ndim != 1 || view.len < 0 ||
                view.len % sizeof(float) != 0 ||
                !PyBuffer_IsContiguous(&view, 'C'))
                throw std::runtime_error("Native FP32 model buffers must be "
                                         "contiguous and one-dimensional");
            Value::FloatArray values(static_cast<std::size_t>(view.len) /
                                     sizeof(float));
            if (view.len)
                std::memcpy(values.data(), view.buf,
                            static_cast<std::size_t>(view.len));
            Value out = Value::float_array(std::move(values));
            memo.remember(object, out);
            return out;
        }
    }
    if (PyDict_Check(object)) {
        Value out = Value::object();
        Py_ssize_t position = 0;
        PyObject *key, *value;
        while (PyDict_Next(object, &position, &key, &value)) {
            if (!PyUnicode_Check(key))
                throw std::runtime_error(
                    "Native boundary requires string dictionary keys");
            std::string name = from_python(key, memo).str();
            out[name] = from_python(value, memo);
        }
        memo.remember(object, out);
        return out;
    }
    PyRef sequence(
        PySequence_Fast(object, "Expected supported native boundary data"));
    if (!sequence.value)
        throw std::runtime_error("Unsupported native boundary object");
    Value out = Value::array();
    out.items().reserve(PySequence_Fast_GET_SIZE(sequence.value));
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(sequence.value); ++i)
        out.push(
            from_python(PySequence_Fast_GET_ITEM(sequence.value, i), memo));
    memo.remember(object, out);
    return out;
}
Value from_python(PyObject *object) {
    FromMemo memo;
    return from_python(object, memo);
}
struct ToMemo {
    std::unordered_map<const void *, PyObject *> values;
    ~ToMemo() {
        for (const auto &[key, object] : values)
            Py_DECREF(object);
    }
    void remember(const void *key, PyObject *object) {
        values.emplace(key, object);
        Py_INCREF(object);
    }
};
PyObject *to_python(const Value &value, ToMemo &memo) {
    if (value.null())
        Py_RETURN_NONE;
    if (auto p = std::get_if<bool>(&value.data))
        return PyBool_FromLong(*p);
    if (auto p = std::get_if<std::int64_t>(&value.data))
        return PyLong_FromLongLong(*p);
    if (auto p = std::get_if<std::uint64_t>(&value.data))
        return PyLong_FromUnsignedLongLong(*p);
    if (auto p = std::get_if<double>(&value.data))
        return PyFloat_FromDouble(*p);
    if (auto p = std::get_if<std::string>(&value.data))
        return PyUnicode_FromStringAndSize(p->data(), p->size());
    const void *identity = nullptr;
    if (auto p = std::get_if<Value::ArrayPtr>(&value.data))
        identity = p->get();
    else if (auto p = std::get_if<Value::FloatArrayPtr>(&value.data))
        identity = p->get();
    else
        identity = std::get<Value::ObjectPtr>(value.data).get();
    auto cached = memo.values.find(identity);
    if (cached != memo.values.end()) {
        Py_INCREF(cached->second);
        return cached->second;
    }
    RecursionGuard recursion;
    if (auto holder = std::get_if<Value::FloatArrayPtr>(&value.data)) {
        const auto &array = **holder;
        PyRef out(PyList_New(array.size()));
        if (!out.value)
            return nullptr;
        memo.remember(identity, out.value);
        for (std::size_t i = 0; i < array.size(); ++i) {
            PyRef item(PyFloat_FromDouble(static_cast<double>(array[i])));
            if (!item.value)
                return nullptr;
            PyList_SET_ITEM(out.value, i, item.release());
        }
        return out.release();
    }
    if (auto holder = std::get_if<Value::ArrayPtr>(&value.data)) {
        const auto &array = **holder;
        PyRef out(PyList_New(array.size()));
        if (!out.value)
            return nullptr;
        memo.remember(identity, out.value);
        for (std::size_t i = 0; i < array.size(); ++i) {
            PyRef item(to_python(array[i], memo));
            if (!item.value)
                return nullptr;
            PyList_SET_ITEM(out.value, i, item.release());
        }
        return out.release();
    }
    PyRef out(PyDict_New());
    if (!out.value)
        return nullptr;
    memo.remember(identity, out.value);
    for (const auto &[key, item] : value.fields()) {
        PyRef name(PyUnicode_FromStringAndSize(key.data(), key.size()));
        if (!name.value)
            return nullptr;
        PyRef converted(to_python(item, memo));
        if (!converted.value ||
            PyDict_SetItem(out.value, name.value, converted.value) < 0)
            return nullptr;
    }
    return out.release();
}
PyObject *to_python(const Value &value) {
    ToMemo memo;
    return to_python(value, memo);
}
template <class T> void destroy(PyObject *capsule) {
    delete static_cast<T *>(PyCapsule_GetPointer(capsule, "wtf.native.ppo.v1"));
}
PPO &ppo(PyObject *capsule) {
    auto *p =
        static_cast<PPO *>(PyCapsule_GetPointer(capsule, "wtf.native.ppo.v1"));
    if (!p)
        throw std::runtime_error("Invalid native PPO handle");
    return *p;
}
PyObject *ppo_create_impl(PyObject *, PyObject *object) {
    auto p = std::make_unique<PPO>(from_python(object));
    PyObject *c = PyCapsule_New(p.get(), "wtf.native.ppo.v1", destroy<PPO>);
    if (c)
        p.release();
    return c;
}
PyObject *ppo_choose_impl(PyObject *, PyObject *args) {
    PyObject *capsule, *observation;
    int train = 1;
    if (!PyArg_ParseTuple(args, "OO|p", &capsule, &observation, &train))
        return nullptr;
    auto input = numbers(from_python(observation));
    return PyUnicode_FromString(ppo(capsule).choose(input, train).c_str());
}
PyObject *ppo_learn_impl(PyObject *, PyObject *args) {
    PyObject *capsule, *observation, *next;
    const char *action;
    double reward;
    int terminal, truncated;
    if (!PyArg_ParseTuple(args, "OOsdOpp", &capsule, &observation, &action,
                          &reward, &next, &terminal, &truncated))
        return nullptr;
    auto input = numbers(from_python(observation));
    std::vector<double> following;
    if (next != Py_None)
        following = numbers(from_python(next));
    ppo(capsule).learn(input, action, reward,
                       next == Py_None ? nullptr : &following, terminal,
                       truncated);
    Py_RETURN_NONE;
}
PyObject *ppo_checkpoint_impl(PyObject *, PyObject *capsule) {
    return to_python(ppo(capsule).checkpoint());
}
PyObject *ppo_diagnostics_impl(PyObject *, PyObject *args) {
    PyObject *capsule, *observation;
    int activations = 0;
    if (!PyArg_ParseTuple(args, "OO|p", &capsule, &observation, &activations))
        return nullptr;
    return to_python(ppo(capsule).diagnostics(numbers(from_python(observation)),
                                              activations));
}
PyObject *controller_create_impl(PyObject *, PyObject *args) {
    PyObject *configuration, *actions, *base, *individual;
    if (!PyArg_ParseTuple(args, "OOOO", &configuration, &actions, &base,
                          &individual))
        return nullptr;
    auto a = from_python(actions), b = from_python(base),
         i = from_python(individual);
    std::vector<std::string> names;
    std::vector<std::uint32_t> bs, is;
    for (const auto &v : a.items())
        names.push_back(v.str());
    for (const auto &v : b.items()) {
        if (v.u64() > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Invalid native seed word");
        bs.push_back(static_cast<std::uint32_t>(v.u64()));
    }
    for (const auto &v : i.items()) {
        if (v.u64() > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Invalid native seed word");
        is.push_back(static_cast<std::uint32_t>(v.u64()));
    }
    return to_python(
        make_controller_state(from_python(configuration), names, bs, is));
}
PyObject *world_contexts_impl(PyObject *, PyObject *checkpoint) {
    World world(from_python(checkpoint));
    Value result = Value::object();
    for (const auto &[name, context] : central_contexts(world))
        result[name] = values(context);
    return to_python(result);
}
PyObject *world_masks_impl(PyObject *, PyObject *args) {
    PyObject *checkpoint, *actions;
    if (!PyArg_ParseTuple(args, "OO", &checkpoint, &actions))
        return nullptr;
    World world(from_python(checkpoint));
    DecisionContext context(world);
    std::vector<std::string> order;
    Value raw = from_python(actions);
    if (actions != Py_None)
        for (const auto &action : raw.items())
            order.push_back(action.str());
    Value result = Value::object();
    for (const auto &animal : static_cast<const World &>(world).animals()) {
        auto selected = order;
        if (actions == Py_None)
            for (const auto &action :
                 world.species(animal).at("allowed_actions").items())
                selected.push_back(action.str());
        Value mask = Value::array();
        for (bool valid : context.action_mask(animal, selected))
            mask.push(valid);
        result[std::to_string(animal.integer("id"))] = std::move(mask);
    }
    return to_python(result);
}
PyObject *group_collect_impl(PyObject *, PyObject *args) {
    PyObject *checkpoint, *states;
    int focal;
    if (!PyArg_ParseTuple(args, "OOi", &checkpoint, &states, &focal))
        return nullptr;
    World world(from_python(checkpoint));
    std::map<int, std::unique_ptr<Controller>> brains;
    auto source = from_python(states);
    for (const auto &[id, state] : source.fields()) {
        brains[std::stoi(id)] = restore_controller(state);
        brains.at(std::stoi(id))->use_world_inputs();
    }
    auto group = collect_group(world, brains, focal);
    Value saved = Value::object();
    for (const auto &[id, brain] : brains)
        saved[std::to_string(id)] = brain->checkpoint();
    return to_python(object(
        {{"group", group}, {"world", world.checkpoint()}, {"brains", saved}}));
}
PyObject *world_create_impl(PyObject *, PyObject *args) {
    PyObject *configuration, *seed;
    if (!PyArg_ParseTuple(args, "OO", &configuration, &seed))
        return nullptr;
    return to_python(
        make_world(from_python(configuration), from_python(seed).u64()));
}
PyObject *world_step_impl(PyObject *, PyObject *args) {
    PyObject *checkpoint, *actions;
    int make_observations = 1;
    if (!PyArg_ParseTuple(args, "OO|p", &checkpoint, &actions,
                          &make_observations))
        return nullptr;
    World world(from_python(checkpoint));
    std::map<int, std::string> chosen;
    Py_ssize_t pos = 0;
    PyObject *id, *action;
    if (!PyDict_Check(actions))
        throw std::runtime_error("Expected action mapping");
    while (PyDict_Next(actions, &pos, &id, &action)) {
        int key;
        if (PyLong_Check(id)) {
            long number = PyLong_AsLong(id);
            if (PyErr_Occurred() || number < 0 || number > 2147483647L)
                throw std::runtime_error("Invalid native animal ID");
            key = static_cast<int>(number);
        } else if (PyUnicode_Check(id)) {
            const char *text = PyUnicode_AsUTF8(id);
            if (!text)
                throw std::runtime_error("Invalid animal ID text");
            std::size_t consumed = 0;
            key = std::stoi(text, &consumed);
            if (consumed != std::string(text).size() || key < 0)
                throw std::runtime_error("Invalid animal ID text");
        } else
            throw std::runtime_error(
                "Native animal IDs must be integers or integer strings");
        if (PyErr_Occurred())
            throw std::runtime_error("Invalid animal ID");
        chosen[key] = from_python(action).str();
    }
    auto step = world.advance(chosen);
    auto observations = make_observations
                            ? observe(world)
                            : std::map<int, std::vector<double>>{};
    Value obs = Value::object(), rewards = Value::object(),
          dead = Value::array();
    for (const auto &[id, o] : observations)
        obs[std::to_string(id)] = values(o);
    for (const auto &[id, r] : step.rewards)
        rewards[std::to_string(id)] = r;
    for (int id : step.dead)
        dead.push(id);
    return to_python(object({{"world", world.checkpoint()},
                             {"observations", obs},
                             {"rewards", rewards},
                             {"dead", dead},
                             {"events", step.events}}));
}
PyObject *observe_impl(PyObject *, PyObject *checkpoint) {
    World world(from_python(checkpoint));
    Value out = Value::object();
    for (const auto &[id, obs] : observe(world))
        out[std::to_string(id)] = values(obs);
    return to_python(
        object({{"observations", out}, {"world", world.checkpoint()}}));
}
PyObject *random_impl(PyObject *, PyObject *args) {
    PyObject *state;
    int count = 10;
    if (!PyArg_ParseTuple(args, "O|i", &state, &count))
        return nullptr;
    if (count < 0 || count > 1000000)
        throw std::runtime_error("Invalid probe count");
    Random rng;
    rng.restore(from_python(state));
    Value rows = Value::array();
    for (int i = 0; i < count; ++i)
        rows.push(rng.random());
    return to_python(object({{"draws", rows}, {"state", rng.checkpoint()}}));
}
struct SessionHandle {
    std::mutex mutex;
    Session engine;
    explicit SessionHandle(const Value &payload) : engine(payload) {}
};
void destroy_session(PyObject *capsule) {
    delete static_cast<SessionHandle *>(
        PyCapsule_GetPointer(capsule, "wtf.native.session.v1"));
}
SessionHandle &session_handle(PyObject *capsule) {
    auto *p = static_cast<SessionHandle *>(
        PyCapsule_GetPointer(capsule, "wtf.native.session.v1"));
    if (!p)
        throw std::runtime_error("Invalid native session handle");
    return *p;
}
struct Lease {
    SessionHandle &handle;
    std::unique_lock<std::mutex> lock;
    explicit Lease(PyObject *capsule)
        : handle(session_handle(capsule)),
          lock(handle.mutex, std::try_to_lock) {
        if (!lock.owns_lock())
            throw std::runtime_error(
                "Native session is busy; query after run returns");
    }
    Session &engine() { return handle.engine; }
};
PyObject *make_session(const Value &payload) {
    auto engine = std::make_unique<SessionHandle>(payload);
    PyObject *handle =
        PyCapsule_New(engine.get(), "wtf.native.session.v1", destroy_session);
    if (handle)
        engine.release();
    return handle;
}
PyObject *create_impl(PyObject *, PyObject *payload) {
    return make_session(from_python(payload));
}
PyObject *bootstrap_impl(PyObject *, PyObject *metadata) {
    return make_session(start_native_run(from_python(metadata)));
}
PyObject *start_run_impl(PyObject *, PyObject *metadata) {
    return to_python(start_native_run(from_python(metadata)));
}
PyObject *signature_impl(PyObject *, PyObject *args) {
    PyObject *config, *actions;
    if (!PyArg_ParseTuple(args, "OO", &config, &actions))
        return nullptr;
    Value raw = from_python(actions);
    std::vector<std::string> names;
    for (const auto &action : raw.items())
        names.push_back(action.str());
    return to_python(native_model_signature(from_python(config), names));
}
PyObject *next_run_impl(PyObject *, PyObject *args) {
    PyObject *capsule, *artifacts = Py_None;
    if (!PyArg_ParseTuple(args, "O|O", &capsule, &artifacts))
        return nullptr;
    Lease lease(capsule);
    auto state = lease.engine().checkpoint();
    if (!state.flag("finished_run") ||
        state.integer("run_index") + 1 >=
            state.at("config").at("experiment").integer("runs") ||
        state.get("early_stopping").flag("stopped"))
        Py_RETURN_NONE;
    if (artifacts != Py_None)
        state["champion_artifacts"] = from_python(artifacts);
    state["run_index"] = state.integer("run_index") + 1;
    return make_session(start_native_run(std::move(state)));
}
PyObject *run_impl(PyObject *, PyObject *args) {
    PyObject *capsule, *options = Py_None;
    int steps = -1;
    const char *path = "";
    if (!PyArg_ParseTuple(args, "O|isO", &capsule, &steps, &path, &options))
        return nullptr;
    Lease lease(capsule);
    Session &engine = lease.engine();
    Value settings =
        options == Py_None ? Value::object() : from_python(options);
    Value result;
    std::exception_ptr error;
    PyThreadState *thread = PyEval_SaveThread();
    try {
        result = engine.run(steps, path, settings);
    } catch (...) {
        error = std::current_exception();
    }
    PyEval_RestoreThread(thread);
    if (error)
        std::rethrow_exception(error);
    return to_python(result);
}
PyObject *checkpoint_impl(PyObject *, PyObject *capsule) {
    Lease lease(capsule);
    return to_python(lease.engine().checkpoint());
}
PyObject *observations_impl(PyObject *, PyObject *capsule) {
    Lease lease(capsule);
    return to_python(lease.engine().observations());
}
PyObject *snapshot_impl(PyObject *, PyObject *args) {
    PyObject *capsule;
    int diagnostics = 0, activations = 0;
    if (!PyArg_ParseTuple(args, "O|pp", &capsule, &diagnostics, &activations))
        return nullptr;
    Lease lease(capsule);
    return to_python(lease.engine().snapshot(diagnostics, activations));
}

constexpr const char *policy_tag = "wtf.native.controller.v1";
void destroy_policy(PyObject *capsule) {
    delete static_cast<Controller *>(PyCapsule_GetPointer(capsule, policy_tag));
}
Controller &policy(PyObject *capsule) {
    auto *p =
        static_cast<Controller *>(PyCapsule_GetPointer(capsule, policy_tag));
    if (!p)
        throw std::runtime_error("Invalid native controller handle");
    return *p;
}
PyObject *policy_create_impl(PyObject *, PyObject *object) {
    auto owner = restore_controller(from_python(object));
    auto *capsule = PyCapsule_New(owner.get(), policy_tag, destroy_policy);
    if (capsule)
        owner.release();
    return capsule;
}
PyObject *policy_choose_impl(PyObject *, PyObject *args) {
    PyObject *handle, *observation;
    int training = 1, random = 0;
    if (!PyArg_ParseTuple(args, "OO|pp", &handle, &observation, &training,
                          &random))
        return nullptr;
    return PyUnicode_FromString(
        policy(handle)
            .choose(numbers(from_python(observation)), training, random)
            .c_str());
}
PyObject *policy_learn_impl(PyObject *, PyObject *args) {
    PyObject *handle, *observation, *next;
    const char *action;
    double reward;
    int terminated, truncated;
    if (!PyArg_ParseTuple(args, "OOsdOpp", &handle, &observation, &action,
                          &reward, &next, &terminated, &truncated))
        return nullptr;
    auto current = numbers(from_python(observation));
    std::vector<double> following;
    if (next != Py_None)
        following = numbers(from_python(next));
    policy(handle).learn(current, action, reward,
                         next == Py_None ? nullptr : &following, terminated,
                         truncated);
    Py_RETURN_NONE;
}
PyObject *policy_checkpoint_impl(PyObject *, PyObject *handle) {
    return to_python(policy(handle).checkpoint());
}
PyObject *policy_diagnostics_impl(PyObject *, PyObject *args) {
    PyObject *handle, *observation;
    int activations = 0;
    if (!PyArg_ParseTuple(args, "OO|p", &handle, &observation, &activations))
        return nullptr;
    return to_python(policy(handle).diagnostics(
        numbers(from_python(observation)), activations));
}
PyObject *policy_model_impl(PyObject *, PyObject *handle) {
    return to_python(policy(handle).model_state());
}
PyObject *policy_world_impl(PyObject *, PyObject *handle) {
    policy(handle).use_world_inputs();
    Py_RETURN_NONE;
}
PyObject *policy_group_impl(PyObject *, PyObject *args) {
    PyObject *handle, *trajectories;
    if (!PyArg_ParseTuple(args, "OO", &handle, &trajectories))
        return nullptr;
    policy(handle).update_group(from_python(trajectories));
    Py_RETURN_NONE;
}
PyObject *policy_context_impl(PyObject *, PyObject *args) {
    PyObject *handle, *context;
    if (!PyArg_ParseTuple(args, "OO", &handle, &context))
        return nullptr;
    policy(handle).set_context(from_python(context));
    Py_RETURN_NONE;
}
PyObject *policy_logits_impl(PyObject *, PyObject *args) {
    PyObject *handle, *observation;
    if (!PyArg_ParseTuple(args, "OO", &handle, &observation))
        return nullptr;
    return to_python(values(
        policy(handle).policy_logits(numbers(from_python(observation)))));
}
PyObject *policy_trial_impl(PyObject *, PyObject *args) {
    PyObject *handle, *observation, *state, *memory = Py_None,
                                            *context = Py_None;
    if (!PyArg_ParseTuple(args, "OOO|OO", &handle, &observation, &state,
                          &memory, &context))
        return nullptr;
    auto &controller = policy(handle);
    Random random;
    random.restore(from_python(state));
    Value hidden =
        memory == Py_None ? controller.trial_memory() : from_python(memory);
    auto action = controller.trial_action(numbers(from_python(observation)),
                                          random, hidden, from_python(context));
    return to_python(object({{"action", action},
                             {"rng", random.checkpoint()},
                             {"memory", hidden}}));
}
PyObject *policy_inherit_impl(PyObject *, PyObject *args) {
    PyObject *mother, *father, *seed;
    int crossover = 1;
    if (!PyArg_ParseTuple(args, "OOO|p", &mother, &father, &seed, &crossover))
        return nullptr;
    std::vector<std::uint32_t> words;
    Value seed_values = from_python(seed);
    for (const auto &word : seed_values.items()) {
        auto value = word.u64();
        if (value > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Invalid seed word");
        words.push_back(static_cast<std::uint32_t>(value));
    }
    return to_python(inherit_controller_state(
        from_python(mother), from_python(father), words, crossover));
}
PyObject *layout_impl(PyObject *, PyObject *configuration) {
    ObservationLayout layout(from_python(configuration));
    auto strings = [](const std::vector<std::string> &source) {
        Value out = Value::array();
        for (const auto &s : source)
            out.push(s);
        return out;
    };
    auto indices = [](const std::vector<std::size_t> &source) {
        Value out = Value::array();
        for (auto i : source)
            out.push(static_cast<std::int64_t>(i));
        return out;
    };
    Value fields = Value::array();
    for (const auto &f : layout.fields) {
        Value names = Value::array();
        for (std::size_t i = f.offset; i < f.offset + f.width; ++i)
            names.push(layout.names.at(i));
        fields.push(object({{"raw", static_cast<std::int64_t>(f.raw)},
                            {"count", static_cast<std::int64_t>(f.count)},
                            {"offset", static_cast<std::int64_t>(f.offset)},
                            {"label", f.label},
                            {"kind", f.kind},
                            {"knots", values(f.knots)},
                            {"names", names}}));
    }
    return to_python(object(
        {{"names", strings(layout.names)},
         {"raw_names", strings(layout.raw_names)},
         {"width", static_cast<std::int64_t>(layout.width())},
         {"physical_width", static_cast<std::int64_t>(layout.physical_width)},
         {"consequence_start",
          static_cast<std::int64_t>(layout.consequence_start)},
         {"core_indices", indices(layout.core_indices)},
         {"confidence_offsets", indices(layout.confidence_offsets)},
         {"fields", fields},
         {"zeroes", values(layout.zeroes)}}));
}
PyObject *soft_encode_impl(PyObject *, PyObject *args) {
    PyObject *configuration, *input;
    if (!PyArg_ParseTuple(args, "OO", &configuration, &input))
        return nullptr;
    ObservationLayout layout(from_python(configuration));
    Value original = from_python(input);
    auto result = layout.encode_soft(numbers(original));
    Value out = values(result);
    for (const auto &field : layout.fields)
        if (field.kind == "raw")
            out.items()[field.offset] = original.items().at(field.raw);
    return to_python(out);
}
PyObject *soft_decode_impl(PyObject *, PyObject *args) {
    PyObject *configuration, *input;
    Py_ssize_t start = 0;
    if (!PyArg_ParseTuple(args, "OO|n", &configuration, &input, &start))
        return nullptr;
    if (start < 0)
        throw std::runtime_error("Negative semantic raw boundary");
    ObservationLayout layout(from_python(configuration));
    Value original = from_python(input);
    auto result =
        layout.decode_soft(numbers(original), static_cast<std::size_t>(start));
    Value out = values(result);
    std::size_t cursor = 0;
    for (const auto &field : layout.fields)
        if (field.raw >= static_cast<std::size_t>(start)) {
            if (field.kind == "raw")
                out.items().at(cursor) = original.items().at(field.offset);
            cursor += field.count;
        }
    return to_python(out);
}
template <PyObject *(*F)(PyObject *, PyObject *)>
PyObject *guard(PyObject *self, PyObject *args) noexcept {
    try {
        return F(self, args);
    } catch (const std::bad_alloc &) {
        return PyErr_NoMemory();
    } catch (const std::exception &e) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
}
PyMethodDef methods[] = {
    {"bootstrap", guard<bootstrap_impl>, METH_O,
     "Create native world and private controllers from metadata."},
    {"start_native_run", guard<start_run_impl>, METH_O,
     "Verify exact native episode initialization."},
    {"model_signature", guard<signature_impl>, METH_VARARGS,
     "Verify native model compatibility metadata."},
    {"next_run", guard<next_run_impl>, METH_VARARGS,
     "Advance native episode scheduling and champion inheritance."},
    {"world_contexts", guard<world_contexts_impl>, METH_O,
     "Verify centralized critic inputs."},
    {"world_masks", guard<world_masks_impl>, METH_VARARGS,
     "Verify physical action masks."},
    {"group_collect", guard<group_collect_impl>, METH_VARARGS,
     "Verify isolated native trial trajectories."},
    {"controller_create_state", guard<controller_create_impl>, METH_VARARGS,
     "Verify native controller genesis."},
    {"world_create", guard<world_create_impl>, METH_VARARGS,
     "Verify native world genesis."},
    {"policy_inherit", guard<policy_inherit_impl>, METH_VARARGS,
     "Verify exact independent newborn controller state."},
    {"policy_create", guard<policy_create_impl>, METH_O,
     "Import a private native controller."},
    {"policy_choose", guard<policy_choose_impl>, METH_VARARGS,
     "Choose through native policy dispatch."},
    {"policy_learn", guard<policy_learn_impl>, METH_VARARGS,
     "Verify independent native learning."},
    {"policy_checkpoint", guard<policy_checkpoint_impl>, METH_O,
     "Export exact private policy state."},
    {"policy_diagnostics", guard<policy_diagnostics_impl>, METH_VARARGS,
     "Collect native per-agent predictions and activations."},
    {"policy_model", guard<policy_model_impl>, METH_O,
     "Export detached model parameters."},
    {"policy_world", guard<policy_world_impl>, METH_O,
     "Enable canonical world observation typing."},
    {"policy_group", guard<policy_group_impl>, METH_VARARGS,
     "Verify native GRPO groups."},
    {"policy_context", guard<policy_context_impl>, METH_VARARGS,
     "Set native policy context."},
    {"policy_logits", guard<policy_logits_impl>, METH_VARARGS,
     "Inspect grouped actor logits."},
    {"policy_trial", guard<policy_trial_impl>, METH_VARARGS,
     "Sample a frozen native trial controller."},
    {"layout", guard<layout_impl>, METH_O,
     "Inspect immutable native observation metadata."},
    {"soft_encode", guard<soft_encode_impl>, METH_VARARGS,
     "Verify exact native V6 projection."},
    {"soft_decode", guard<soft_decode_impl>, METH_VARARGS,
     "Verify exact native V6 target decoding."},
    {"create", guard<create_impl>, METH_O,
     "Import one native-owned episode session."},
    {"run", guard<run_impl>, METH_VARARGS,
     "Run native world, sensing and private learners with the GIL released."},
    {"checkpoint", guard<checkpoint_impl>, METH_O,
     "Export compatible experiment state on request."},
    {"observations", guard<observations_impl>, METH_O,
     "Query detached observations."},
    {"snapshot", guard<snapshot_impl>, METH_VARARGS,
     "Query a detached world snapshot."},
    {"ppo_create", guard<ppo_create_impl>, METH_O,
     "Import an independent native PPO state."},
    {"ppo_choose", guard<ppo_choose_impl>, METH_VARARGS,
     "Query one native policy."},
    {"ppo_learn", guard<ppo_learn_impl>, METH_VARARGS,
     "Supply one transition for parity verification."},
    {"ppo_checkpoint", guard<ppo_checkpoint_impl>, METH_O,
     "Export private controller state."},
    {"ppo_diagnostics", guard<ppo_diagnostics_impl>, METH_VARARGS,
     "Query native predictions."},
    {"world_step", guard<world_step_impl>, METH_VARARGS,
     "Verify one native physical step."},
    {"observe", guard<observe_impl>, METH_O, "Query native observations."},
    {"random_probe", guard<random_impl>, METH_VARARGS,
     "Verify native RNG consumption."},
    {nullptr, nullptr, 0, nullptr}};
PyModuleDef engine_module = {PyModuleDef_HEAD_INIT,
                             "_engine",
                             "Native-owned simulation and learning runtime.",
                             -1,
                             methods,
                             nullptr,
                             nullptr,
                             nullptr,
                             nullptr};
} // namespace
PyMODINIT_FUNC PyInit__engine() { return PyModule_Create(&engine_module); }
