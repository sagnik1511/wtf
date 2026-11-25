// Optional NVCC dense learning backend. FP64 default; explicit FP32 arithmetic.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <new>
#include <vector>

namespace {
thread_local char last_error[512] = "";
int fail(cudaError_t code) {
    if (code == cudaSuccess)
        return 0;
    std::snprintf(last_error, sizeof(last_error), "%s",
                  cudaGetErrorString(code));
    return 1;
}
struct Workspace {
    int models, batch, layers, inputs, outputs, parameters, activations, mode;
    bool fp32, profile, profile_error = false;
    int *sizes = nullptr, *po = nullptr, *ao = nullptr;
    int *actions = nullptr, *valid = nullptr, *metadata = nullptr;
    unsigned char *masks = nullptr;
    float *weights = nullptr, *targets = nullptr, *moments = nullptr,
          *variances = nullptr;
    void *observations = nullptr, *next = nullptr, *rewards = nullptr;
    void *scratch = nullptr, *metrics = nullptr, *output = nullptr,
         *summary = nullptr;
    cudaEvent_t start = nullptr, finish = nullptr;
    double stats[8] = {};
    std::vector<float> cached_weights, cached_targets, cached_moments,
        cached_variances;
    ~Workspace() {
        cudaFree(sizes);
        cudaFree(po);
        cudaFree(ao);
        cudaFree(actions);
        cudaFree(valid);
        cudaFree(metadata);
        cudaFree(masks);
        cudaFree(weights);
        cudaFree(targets);
        cudaFree(moments);
        cudaFree(variances);
        cudaFree(observations);
        cudaFree(next);
        cudaFree(rewards);
        cudaFree(scratch);
        cudaFree(metrics);
        cudaFree(output);
        cudaFree(summary);
        if (start)
            cudaEventDestroy(start);
        if (finish)
            cudaEventDestroy(finish);
    }
};
struct Timer {
    Workspace *w;
    int stage;
    Timer(Workspace *workspace, int index) : w(workspace), stage(index) {
        if (w->profile && fail(cudaEventRecord(w->start)))
            w->profile_error = true;
    }
    ~Timer() {
        if (!w->profile)
            return;
        float elapsed = 0;
        if (fail(cudaEventRecord(w->finish)) ||
            fail(cudaEventSynchronize(w->finish)) ||
            fail(cudaEventElapsedTime(&elapsed, w->start, w->finish)))
            w->profile_error = true;
        else
            w->stats[stage] += elapsed * .001;
    }
};
bool allocate(void **p, size_t bytes) { return !fail(cudaMalloc(p, bytes)); }
template <class T> bool allocate(T **p, size_t n) {
    return allocate(reinterpret_cast<void **>(p), n * sizeof(T));
}
bool upload(Workspace *w, void *device, const void *host, size_t bytes) {
    if (fail(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice)))
        return false;
    w->stats[3] += static_cast<double>(bytes);
    return true;
}
bool download(Workspace *w, void *host, const void *device, size_t bytes) {
    if (fail(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost)))
        return false;
    w->stats[4] += static_cast<double>(bytes);
    return true;
}
bool cached_upload(Workspace *w, float *device, const float *host,
                   std::vector<float> &cache, bool nonnegative = false,
                   bool count_weights = false) {
    size_t n = static_cast<size_t>(w->models) * w->parameters;
    if (cache.size() == n &&
        std::memcmp(cache.data(), host, n * sizeof(float)) == 0) {
        if (count_weights)
            w->stats[6] += 1;
        return true;
    }
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(host[i]) || (nonnegative && host[i] < 0)) {
            std::snprintf(last_error, sizeof(last_error),
                          "Invalid CUDA parameter/optimizer state");
            return false;
        }
    if (!upload(w, device, host, n * sizeof(float)))
        return false;
    try {
        cache.assign(host, host + n);
    } catch (const std::bad_alloc &) {
        std::snprintf(last_error, sizeof(last_error),
                      "Host CUDA weight cache allocation failed");
        return false;
    }
    if (count_weights)
        w->stats[5] += 1;
    return true;
}
bool weights(Workspace *w, const float *host, bool target = false) {
    return cached_upload(w, target ? w->targets : w->weights, host,
                         target ? w->cached_targets : w->cached_weights, false,
                         true);
}
template <class R> bool finite_output(const void *pointer, size_t count) {
    const R *values = static_cast<const R *>(pointer);
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) {
            std::snprintf(last_error, sizeof(last_error),
                          "Non-finite CUDA output");
            return false;
        }
    return true;
}
// One warp cooperates on each neuron: adjacent lanes read adjacent weights.
template <class R>
__device__ void forward(const float *weights, const R *input, R *tape,
                        const int *sizes, const int *po, const int *ao,
                        int layers) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int i = threadIdx.x; i < sizes[0]; i += blockDim.x)
        tape[i] = input[i];
    __syncthreads();
    for (int l = 0; l < layers; ++l) {
        int ni = sizes[l], no = sizes[l + 1];
        for (int j = warp; j < no; j += blockDim.x / 32) {
            R value = 0;
            for (int i = lane; i < ni; i += 32)
                value += tape[ao[l] + i] * R(weights[po[l] + j * ni + i]);
            for (int offset = 16; offset; offset /= 2)
                value += __shfl_down_sync(0xffffffff, value, offset);
            if (!lane) {
                value += R(weights[po[l] + ni * no + j]);
                tape[ao[l + 1] + j] = l + 1 == layers ? value : tanh(value);
            }
        }
        __syncthreads();
    }
}
// Save activation/output deltas only, not B copies of the full parameter
// gradient.
template <class R>
__device__ void backward(const float *weights, const R *tape, R *delta,
                         const int *sizes, const int *po, const int *ao,
                         int layers) {
    for (int l = layers - 1; l > 0; --l) {
        int ni = sizes[l], no = sizes[l + 1];
        for (int i = threadIdx.x; i < ni; i += blockDim.x) {
            R total = 0;
            for (int j = 0; j < no; ++j)
                total += R(weights[po[l] + j * ni + i]) * delta[ao[l + 1] + j];
            R value = tape[ao[l] + i];
            delta[ao[l] + i] = total * (R(1) - value * value);
        }
        __syncthreads();
    }
}
template <class R>
__global__ void
dqn_samples(const float *weights, const float *targets, const R *obs,
            const R *next, const R *rewards, const int *actions,
            const int *terminal, const int *sizes, const int *po, const int *ao,
            int layers, int parameters, int activations, int batch, R discount,
            R threshold, int double_q, R *scratch, R *metrics) {
    int sample = blockIdx.x, model = sample / batch, inputs = sizes[0];
    const float *online = weights + static_cast<size_t>(model) * parameters;
    R *tape = scratch + static_cast<size_t>(sample) * 2 * activations,
      *delta = tape + activations;
    __shared__ R target;
    __shared__ int best;
    forward(online, obs + static_cast<size_t>(sample) * inputs, tape, sizes, po,
            ao, layers);
    if (!terminal[sample]) {
        if (double_q) {
            forward(online, next + static_cast<size_t>(sample) * inputs, delta,
                    sizes, po, ao, layers);
            if (!threadIdx.x) {
                best = 0;
                for (int j = 1; j < sizes[layers]; ++j)
                    if (delta[ao[layers] + j] > delta[ao[layers] + best])
                        best = j;
            }
            __syncthreads();
        }
        forward(targets + static_cast<size_t>(model) * parameters,
                next + static_cast<size_t>(sample) * inputs, delta, sizes, po,
                ao, layers);
        if (!threadIdx.x) {
            if (!double_q) {
                best = 0;
                for (int j = 1; j < sizes[layers]; ++j)
                    if (delta[ao[layers] + j] > delta[ao[layers] + best])
                        best = j;
            }
            target = rewards[sample] + discount * delta[ao[layers] + best];
        }
    } else if (!threadIdx.x)
        target = rewards[sample];
    __syncthreads();
    R error = tape[ao[layers] + actions[sample]] - target;
    R derivative = max(-threshold, min(threshold, error));
    if (!threadIdx.x) {
        R absolute = abs(error);
        metrics[sample] = absolute <= threshold
                              ? R(.5) * error * error
                              : threshold * (absolute - R(.5) * threshold);
    }
    for (int j = threadIdx.x; j < sizes[layers]; j += blockDim.x)
        delta[ao[layers] + j] = j == actions[sample] ? derivative : R(0);
    __syncthreads();
    backward(online, tape, delta, sizes, po, ao, layers);
}
template <class R>
__global__ void ppo_samples(const float *weights, const R *obs, const R *data,
                            const int *actions, const int *valid,
                            const unsigned char *masks, const int *sizes,
                            const int *po, const int *ao, int layers,
                            int parameters, int activations, int batch, R clip,
                            R entropy_coefficient, R value_coefficient,
                            R *scratch, R *metrics) {
    int sample = blockIdx.x, model = sample / batch, n = sizes[layers] - 1;
    R *tape = scratch + static_cast<size_t>(sample) * 2 * activations,
      *delta = tape + activations;
    const float *online = weights + static_cast<size_t>(model) * parameters;
    if (!valid[sample]) {
        for (int i = threadIdx.x; i < 2 * activations; i += blockDim.x)
            tape[i] = 0;
        for (int i = threadIdx.x; i < 5; i += blockDim.x)
            metrics[sample * 5 + i] = 0;
        return;
    }
    forward(online, obs + static_cast<size_t>(sample) * sizes[0], tape, sizes,
            po, ao, layers);
    if (!threadIdx.x) {
        R *out = tape + ao[layers], *grad = delta + ao[layers];
        const unsigned char *mask = masks + static_cast<size_t>(sample) * n;
        R maximum = mask[0] ? out[0] : R(-1e30), sum = 0;
        for (int j = 1; j < n; ++j)
            maximum = max(maximum, mask[j] ? out[j] : R(-1e30));
        for (int j = 0; j < n; ++j)
            sum += exp((mask[j] ? out[j] : R(-1e30)) - maximum);
        R lognorm = log(sum), entropy = 0;
        for (int j = 0; j < n; ++j) {
            R lp = ((mask[j] ? out[j] : R(-1e30)) - maximum) - lognorm;
            entropy -= exp(lp) * lp;
        }
        R old = data[sample * 3], advantage = data[sample * 3 + 1],
          target = data[sample * 3 + 2];
        R lp = ((mask[actions[sample]] ? out[actions[sample]] : R(-1e30)) -
                maximum) -
               lognorm;
        R logratio = max(R(-60), min(R(60), lp - old)), ratio = exp(logratio);
        bool clipped = (advantage > 0 && ratio > R(1) + clip) ||
                       (advantage < 0 && ratio < R(1) - clip);
        R derivative = clipped ? R(0) : -advantage * ratio;
        for (int j = 0; j < n; ++j) {
            R logp = ((mask[j] ? out[j] : R(-1e30)) - maximum) - lognorm,
              p = exp(logp);
            grad[j] = mask[j] ? derivative * (R(j == actions[sample]) - p) +
                                    entropy_coefficient * p * (logp + entropy)
                              : R(0);
        }
        R error = out[n] - target;
        grad[n] = value_coefficient * error;
        metrics[sample * 5] =
            -min(ratio * advantage,
                 max(R(1) - clip, min(R(1) + clip, ratio)) * advantage);
        metrics[sample * 5 + 1] = R(.5) * error * error;
        metrics[sample * 5 + 2] = entropy;
        metrics[sample * 5 + 3] = ratio - R(1) - logratio;
        metrics[sample * 5 + 4] = R(abs(ratio - R(1)) > clip);
    }
    __syncthreads();
    backward(online, tape, delta, sizes, po, ao, layers);
}
template <class R>
__global__ void reduce_gradients(const R *scratch, const R *metrics, R *output,
                                 const int *sizes, const int *po, const int *ao,
                                 int layers, int parameters, int activations,
                                 int batch, int models, int metric_count) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(models) * (parameters + metric_count))
        return;
    int model = index / (parameters + metric_count),
        p = index % (parameters + metric_count), layer = 0;
    if (p < parameters)
        while (layer + 1 < layers && p >= po[layer + 1])
            ++layer;
    int local = p < parameters ? p - po[layer] : 0, ni = sizes[layer],
        no = sizes[layer + 1];
    R total = 0;
    for (int b = 0; b < batch; ++b) {
        size_t sample = static_cast<size_t>(model) * batch + b;
        if (p >= parameters)
            total += metrics[sample * metric_count + p - parameters];
        else {
            const R *tape = scratch + sample * 2 * activations,
                    *delta = tape + activations;
            total += local < ni * no ? delta[ao[layer + 1] + local / ni] *
                                           tape[ao[layer] + local % ni]
                                     : delta[ao[layer + 1] + local - ni * no];
        }
    }
    output[index] = total;
}
template <class R>
__global__ void adam(float *weights, float *moments, float *variances,
                     const R *output, R *summary, const int *meta,
                     int parameters, R norm_limit, R rate, R kl_limit) {
    int model = blockIdx.x, count = meta[model * 2], step = meta[model * 2 + 1];
    const R *gradient = output + static_cast<size_t>(model) * (parameters + 5);
    R *report = summary + model * 6;
    bool accepted =
        count > 0 &&
        (!kl_limit || gradient[parameters + 3] / R(count) <= kl_limit);
    for (int j = threadIdx.x; j < 5; j += blockDim.x)
        report[j] = gradient[parameters + j];
    if (!threadIdx.x)
        report[5] = R(accepted);
    if (!accepted)
        return;
    __shared__ R sums[128];
    R norm = 0;
    for (int p = threadIdx.x; p < parameters; p += blockDim.x) {
        R v = gradient[p] / R(count);
        norm += v * v;
    }
    sums[threadIdx.x] = norm;
    __syncthreads();
    for (int offset = 64; offset; offset /= 2) {
        if (threadIdx.x < offset)
            sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    R factor = min(R(1), norm_limit / (sqrt(sums[0]) + R(1e-12))) / R(count);
    R correction1 = R(1) - pow(R(.9), R(step)),
      correction2 = R(1) - pow(R(.999), R(step));
    for (int p = threadIdx.x; p < parameters; p += blockDim.x) {
        size_t at = static_cast<size_t>(model) * parameters + p;
        R g = gradient[p] * factor;
        float m = R(.9) * R(moments[at]) + R(.1) * g;
        float v = R(.999) * R(variances[at]) + R(.001) * g * g;
        moments[at] = m;
        variances[at] = v;
        weights[at] = R(weights[at]) - rate * (R(m) / correction1) /
                                           (sqrt(R(v) / correction2) + R(1e-8));
    }
}
template <class R>
__global__ void predictions(const float *weights, const R *obs, R *scratch,
                            R *output, const int *sizes, const int *po,
                            const int *ao, int layers, int parameters,
                            int activations) {
    int model = blockIdx.x;
    R *tape = scratch + static_cast<size_t>(model) * 2 * activations;
    forward(weights + static_cast<size_t>(model) * parameters,
            obs + static_cast<size_t>(model) * sizes[0], tape, sizes, po, ao,
            layers);
    for (int j = threadIdx.x; j < sizes[layers]; j += blockDim.x)
        output[static_cast<size_t>(model) * sizes[layers] + j] =
            tape[ao[layers] + j];
}
template <class R>
__global__ void dqn_adam(float *weights, float *targets, float *moments,
                         float *variances, const R *output, R *losses,
                         const int *metadata, int parameters, int batch,
                         double norm_limit, double rate) {
    int model = blockIdx.x, step = metadata[model * 2];
    const R *gradient = output + static_cast<size_t>(model) * (parameters + 1);
    __shared__ double factor, correction1, correction2;
    if (!threadIdx.x) {
        // The CPU optimizer widens gradients before arithmetic, including FP32
        // CUDA gradients. Preserve that contract and compensated norm ordering.
        double sum = 0, correction = 0;
        for (int p = 0; p < parameters; ++p) {
            double g = double(gradient[p]) / double(batch), x = g * g;
            double next = sum + x;
            correction +=
                abs(sum) >= abs(x) ? (sum - next) + x : (x - next) + sum;
            sum = next;
        }
        double norm =
            sqrt(correction && isfinite(correction) ? sum + correction : sum);
        factor = min(1.0, norm_limit / (norm + 1e-12)) / double(batch);
        correction1 = 1.0 - pow(0.9, double(step));
        correction2 = 1.0 - pow(0.999, double(step));
        losses[model] = gradient[parameters];
    }
    __syncthreads();
    for (int p = threadIdx.x; p < parameters; p += blockDim.x) {
        size_t at = static_cast<size_t>(model) * parameters + p;
        double g = double(gradient[p]) * factor;
        float m = 0.9 * double(moments[at]) + 0.1 * g;
        float v = 0.999 * double(variances[at]) + 0.001 * g * g;
        float weight =
            double(weights[at]) - rate * (double(m) / correction1) /
                                      (sqrt(double(v) / correction2) + 1e-8);
        moments[at] = m;
        variances[at] = v;
        weights[at] = weight;
        if (metadata[model * 2 + 1])
            targets[at] = weight;
    }
}

template <class R>
int run_dqn(Workspace *w, const float *online, const float *target,
            const void *obs, const void *next, const void *rewards,
            const int *actions, const int *terminal, double discount,
            double threshold, int double_q, void *output, bool publish = true) {
    size_t samples = static_cast<size_t>(w->models) * w->batch;
    {
        Timer timer(w, 0);
        if (!weights(w, online) || !weights(w, target, true) ||
            !upload(w, w->observations, obs, samples * w->inputs * sizeof(R)) ||
            !upload(w, w->next, next, samples * w->inputs * sizeof(R)) ||
            !upload(w, w->rewards, rewards, samples * sizeof(R)) ||
            !upload(w, w->actions, actions, samples * sizeof(int)) ||
            !upload(w, w->valid, terminal, samples * sizeof(int)))
            return 1;
    }
    {
        Timer timer(w, 1);
        dqn_samples<<<samples, 128>>>(
            w->weights, w->targets, static_cast<R *>(w->observations),
            static_cast<R *>(w->next), static_cast<R *>(w->rewards), w->actions,
            w->valid, w->sizes, w->po, w->ao, w->layers, w->parameters,
            w->activations, w->batch, R(discount), R(threshold), double_q,
            static_cast<R *>(w->scratch), static_cast<R *>(w->metrics));
        if (fail(cudaGetLastError()))
            return 1;
        size_t n = static_cast<size_t>(w->models) * (w->parameters + 1);
        reduce_gradients<<<(n + 255) / 256, 256>>>(
            static_cast<R *>(w->scratch), static_cast<R *>(w->metrics),
            static_cast<R *>(w->output), w->sizes, w->po, w->ao, w->layers,
            w->parameters, w->activations, w->batch, w->models, 1);
        if (fail(cudaGetLastError()))
            return 1;
    }
    if (publish) {
        Timer timer(w, 2);
        if (!download(w, output, w->output,
                      static_cast<size_t>(w->models) * (w->parameters + 1) *
                          sizeof(R)))
            return 1;
    }
    return w->profile_error ||
           (publish &&
            !finite_output<R>(output, static_cast<size_t>(w->models) *
                                          (w->parameters + 1)));
}

template <class R>
int dqn_update(Workspace *w, float *online, const float *target,
               const void *obs, const void *next, const void *rewards,
               const int *actions, const int *terminal, float *moments,
               float *variances, const int *metadata, const double *options,
               void *losses) {
    for (int model = 0; model < w->models; ++model)
        if (metadata[model * 2] < 1 ||
            (metadata[model * 2 + 1] != 0 && metadata[model * 2 + 1] != 1)) {
            std::snprintf(last_error, sizeof(last_error),
                          "Invalid CUDA DQN update metadata");
            return 1;
        }
    for (int i = 0; i < 5; ++i)
        if (!std::isfinite(options[i])) {
            std::snprintf(last_error, sizeof(last_error),
                          "Invalid CUDA DQN options");
            return 1;
        }
    if (options[0] < 0 || options[0] > 1 || options[1] <= 0 ||
        options[2] <= 0 || options[3] <= 0 ||
        (options[4] != 0 && options[4] != 1)) {
        std::snprintf(last_error, sizeof(last_error),
                      "Invalid CUDA DQN options");
        return 1;
    }
    {
        Timer timer(w, 0);
        if (!cached_upload(w, w->moments, moments, w->cached_moments) ||
            !cached_upload(w, w->variances, variances, w->cached_variances,
                           true) ||
            !upload(w, w->metadata, metadata, w->models * 2 * sizeof(int)))
            return 1;
    }
    if (run_dqn<R>(w, online, target, obs, next, rewards, actions, terminal,
                   options[0], options[1], int(options[4]), nullptr, false))
        return 1;
    {
        Timer timer(w, 1);
        dqn_adam<<<w->models, 128>>>(
            w->weights, w->targets, w->moments, w->variances,
            static_cast<R *>(w->output), static_cast<R *>(w->summary),
            w->metadata, w->parameters, w->batch, options[2], options[3]);
        if (fail(cudaGetLastError()))
            return 1;
    }
    size_t n = static_cast<size_t>(w->models) * w->parameters;
    {
        Timer timer(w, 2);
        if (!download(w, online, w->weights, n * sizeof(float)) ||
            !download(w, moments, w->moments, n * sizeof(float)) ||
            !download(w, variances, w->variances, n * sizeof(float)) ||
            !download(w, losses, w->summary, w->models * sizeof(R)))
            return 1;
    }
    if (w->profile_error || !finite_output<float>(online, n) ||
        !finite_output<float>(moments, n) ||
        !finite_output<float>(variances, n) ||
        !finite_output<R>(losses, w->models))
        return 1;
    // Refresh exact-content host mirrors after publication. Slot reuse with a
    // different brain or arbitrary CPU edits must invalidate all private state.
    std::copy(online, online + n, w->cached_weights.begin());
    std::copy(moments, moments + n, w->cached_moments.begin());
    std::copy(variances, variances + n, w->cached_variances.begin());
    for (int model = 0; model < w->models; ++model)
        if (metadata[model * 2 + 1])
            std::copy(online + static_cast<size_t>(model) * w->parameters,
                      online + static_cast<size_t>(model + 1) * w->parameters,
                      w->cached_targets.begin() +
                          static_cast<size_t>(model) * w->parameters);
    return 0;
}
template <class R>
int predict(Workspace *w, const float *online, const void *obs, void *output) {
    {
        Timer timer(w, 0);
        if (!weights(w, online) ||
            !upload(w, w->observations, obs,
                    static_cast<size_t>(w->models) * w->inputs * sizeof(R)))
            return 1;
    }
    {
        Timer timer(w, 1);
        predictions<<<w->models, 128>>>(
            w->weights, static_cast<R *>(w->observations),
            static_cast<R *>(w->scratch), static_cast<R *>(w->output), w->sizes,
            w->po, w->ao, w->layers, w->parameters, w->activations);
        if (fail(cudaGetLastError()))
            return 1;
    }
    {
        Timer timer(w, 2);
        if (!download(w, output, w->output,
                      static_cast<size_t>(w->models) * w->outputs * sizeof(R)))
            return 1;
    }
    return w->profile_error;
}
template <class R>
int ppo(Workspace *w, const void *obs, const void *data, const int *actions,
        const int *valid, const unsigned char *masks, const int *metadata,
        const double *options, void *summary) {
    size_t samples = static_cast<size_t>(w->models) * w->batch;
    {
        Timer timer(w, 0);
        if (!upload(w, w->observations, obs, samples * w->inputs * sizeof(R)) ||
            !upload(w, w->rewards, data, samples * 3 * sizeof(R)) ||
            !upload(w, w->actions, actions, samples * sizeof(int)) ||
            !upload(w, w->valid, valid, samples * sizeof(int)) ||
            !upload(w, w->masks, masks, samples * (w->outputs - 1)) ||
            !upload(w, w->metadata, metadata, w->models * 2 * sizeof(int)))
            return 1;
    }
    {
        Timer timer(w, 1);
        ppo_samples<<<samples, 128>>>(
            w->weights, static_cast<R *>(w->observations),
            static_cast<R *>(w->rewards), w->actions, w->valid, w->masks,
            w->sizes, w->po, w->ao, w->layers, w->parameters, w->activations,
            w->batch, R(options[0]), R(options[1]), R(options[2]),
            static_cast<R *>(w->scratch), static_cast<R *>(w->metrics));
        if (fail(cudaGetLastError()))
            return 1;
        size_t n = static_cast<size_t>(w->models) * (w->parameters + 5);
        reduce_gradients<<<(n + 255) / 256, 256>>>(
            static_cast<R *>(w->scratch), static_cast<R *>(w->metrics),
            static_cast<R *>(w->output), w->sizes, w->po, w->ao, w->layers,
            w->parameters, w->activations, w->batch, w->models, 5);
        if (fail(cudaGetLastError()))
            return 1;
        adam<<<w->models, 128>>>(
            w->weights, w->moments, w->variances, static_cast<R *>(w->output),
            static_cast<R *>(w->summary), w->metadata, w->parameters,
            R(options[3]), R(options[4]), R(options[5]));
        if (fail(cudaGetLastError()))
            return 1;
        w->cached_weights.clear();
    }
    {
        Timer timer(w, 2);
        if (!download(w, summary, w->summary, w->models * 6 * sizeof(R)))
            return 1;
    }
    return w->profile_error;
}
} // namespace
extern "C" {
int wtf_cuda_api_version() { return 3; }
const char *wtf_cuda_error() { return last_error; }
int wtf_cuda_probe(char *name, int capacity) {
    int device;
    cudaDeviceProp properties;
    if (fail(cudaGetDevice(&device)) ||
        fail(cudaGetDeviceProperties(&properties, device)))
        return 1;
    std::snprintf(name, capacity, "%s", properties.name);
    return 0;
}
void *wtf_cuda_create(const int *sizes, int layers, int models, int batch,
                      int mode, int fp32, int profile) {
    if (layers < 1 || layers > 16 || models < 1 || models > 4096 || batch < 1 ||
        batch > 4096 || mode < 0 || mode > 3 || (mode == 0 && batch != 1) ||
        (fp32 != 0 && fp32 != 1) || (profile != 0 && profile != 1)) {
        std::snprintf(last_error, sizeof(last_error),
                      "Invalid CUDA workspace dimensions/mode");
        return nullptr;
    }
    for (int i = 0; i <= layers; ++i)
        if (sizes[i] < 1 || sizes[i] > 4096) {
            std::snprintf(last_error, sizeof(last_error),
                          "CUDA layer width must be 1..4096");
            return nullptr;
        }
    auto *w = new (std::nothrow) Workspace;
    if (!w) {
        std::snprintf(last_error, sizeof(last_error), "Host allocation failed");
        return nullptr;
    }
    try {
        w->models = models;
        w->batch = batch;
        w->layers = layers;
        w->inputs = sizes[0];
        w->outputs = sizes[layers];
        w->mode = mode;
        w->fp32 = fp32;
        w->profile = profile;
        std::vector<int> po(layers + 1, 0), ao(layers + 2, 0);
        for (int l = 0; l <= layers; ++l) {
            ao[l + 1] = ao[l] + sizes[l];
            if (l < layers)
                po[l + 1] = po[l] + (sizes[l] + 1) * sizes[l + 1];
        }
        w->parameters = po[layers];
        w->activations = ao[layers + 1];
        size_t samples = static_cast<size_t>(models) * batch,
               n = static_cast<size_t>(models) * w->parameters;
        size_t real = fp32 ? sizeof(float) : sizeof(double);
        bool ok =
            allocate(&w->sizes, layers + 1) && allocate(&w->po, layers + 1) &&
            allocate(&w->ao, layers + 2) && allocate(&w->weights, n) &&
            allocate(&w->observations, samples * w->inputs * real) &&
            allocate(&w->scratch, samples * 2 * w->activations * real) &&
            allocate(&w->output, static_cast<size_t>(models) *
                                     (mode ? w->parameters + (mode == 2 ? 5 : 1)
                                           : w->outputs) *
                                     real);
        if (ok && mode)
            ok = allocate(&w->actions, samples) &&
                 allocate(&w->valid, samples) &&
                 allocate(&w->metrics, samples * (mode == 2 ? 5 : 1) * real) &&
                 allocate(&w->rewards, samples * (mode == 2 ? 3 : 1) * real);
        if (ok && (mode == 1 || mode == 3))
            ok = allocate(&w->targets, n) &&
                 allocate(&w->next, samples * w->inputs * real);
        if (ok && mode == 3)
            ok = allocate(&w->moments, n) && allocate(&w->variances, n) &&
                 allocate(&w->metadata, models * 2) &&
                 allocate(&w->summary, models * real);
        if (ok && mode == 2)
            ok = allocate(&w->moments, n) && allocate(&w->variances, n) &&
                 allocate(&w->masks, samples * (w->outputs - 1)) &&
                 allocate(&w->metadata, models * 2) &&
                 allocate(&w->summary, models * 6 * real);
        if (ok && profile)
            ok = !fail(cudaEventCreate(&w->start)) &&
                 !fail(cudaEventCreate(&w->finish));
        if (ok)
            ok = upload(w, w->sizes, sizes, (layers + 1) * sizeof(int)) &&
                 upload(w, w->po, po.data(), (layers + 1) * sizeof(int)) &&
                 upload(w, w->ao, ao.data(), (layers + 2) * sizeof(int));
        if (!ok) {
            delete w;
            return nullptr;
        }
        return w;
    } catch (const std::bad_alloc &) {
        delete w;
        std::snprintf(last_error, sizeof(last_error),
                      "Host CUDA workspace allocation failed");
        return nullptr;
    }
}
void wtf_cuda_destroy(void *p) { delete static_cast<Workspace *>(p); }
int wtf_cuda_stats(void *p, double *out) {
    std::memcpy(out, static_cast<Workspace *>(p)->stats, 8 * sizeof(double));
    return 0;
}
int wtf_cuda_run(void *p, const float *weights, const float *targets,
                 const void *obs, const void *next, const void *rewards,
                 const int *actions, const int *terminal, double discount,
                 double threshold, int double_q, void *output) {
    auto *w = static_cast<Workspace *>(p);
    if (w->mode != 1) {
        std::snprintf(last_error, sizeof(last_error), "DQN workspace required");
        return 1;
    }
    return w->fp32 ? run_dqn<float>(w, weights, targets, obs, next, rewards,
                                    actions, terminal, discount, threshold,
                                    double_q, output)
                   : run_dqn<double>(w, weights, targets, obs, next, rewards,
                                     actions, terminal, discount, threshold,
                                     double_q, output);
}
int wtf_cuda_predict(void *p, const float *weights, const void *obs,
                     void *output) {
    auto *w = static_cast<Workspace *>(p);
    if (w->mode) {
        std::snprintf(last_error, sizeof(last_error),
                      "Prediction workspace required");
        return 1;
    }
    return w->fp32 ? predict<float>(w, weights, obs, output)
                   : predict<double>(w, weights, obs, output);
}
int wtf_cuda_dqn_update(void *p, float *weights, const float *targets,
                        const void *obs, const void *next, const void *rewards,
                        const int *actions, const int *terminal, float *moments,
                        float *variances, const int *metadata,
                        const double *options, void *losses) {
    auto *w = static_cast<Workspace *>(p);
    if (w->mode != 3) {
        std::snprintf(last_error, sizeof(last_error),
                      "Resident DQN workspace required");
        return 1;
    }
    int result = w->fp32
                     ? dqn_update<float>(w, weights, targets, obs, next,
                                         rewards, actions, terminal, moments,
                                         variances, metadata, options, losses)
                     : dqn_update<double>(w, weights, targets, obs, next,
                                          rewards, actions, terminal, moments,
                                          variances, metadata, options, losses);
    if (result) {
        // A failed publication may have left mutated device slots. Force fresh
        // uploads on retry rather than treating stale CPU snapshots as cache
        // hits.
        w->cached_weights.clear();
        w->cached_targets.clear();
        w->cached_moments.clear();
        w->cached_variances.clear();
    }
    return result;
}
int wtf_cuda_ppo_begin(void *p, const float *online, const float *moments,
                       const float *variances) {
    auto *w = static_cast<Workspace *>(p);
    if (w->mode != 2) {
        std::snprintf(last_error, sizeof(last_error), "PPO workspace required");
        return 1;
    }
    size_t count = static_cast<size_t>(w->models) * w->parameters;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(moments[i]) || !std::isfinite(variances[i]) ||
            variances[i] < 0) {
            std::snprintf(last_error, sizeof(last_error),
                          "Invalid CUDA PPO Adam state");
            return 1;
        }
    {
        Timer timer(w, 0);
        size_t bytes = count * sizeof(float);
        if (!weights(w, online) || !upload(w, w->moments, moments, bytes) ||
            !upload(w, w->variances, variances, bytes))
            return 1;
    }
    return w->profile_error;
}
int wtf_cuda_ppo(void *p, const void *obs, const void *data, const int *actions,
                 const int *valid, const unsigned char *masks,
                 const int *metadata, const double *options, void *summary) {
    auto *w = static_cast<Workspace *>(p);
    if (w->mode != 2) {
        std::snprintf(last_error, sizeof(last_error), "PPO workspace required");
        return 1;
    }
    return w->fp32 ? ppo<float>(w, obs, data, actions, valid, masks, metadata,
                                options, summary)
                   : ppo<double>(w, obs, data, actions, valid, masks, metadata,
                                 options, summary);
}
int wtf_cuda_ppo_end(void *p, float *online, float *moments, float *variances) {
    auto *w = static_cast<Workspace *>(p);
    if (w->mode != 2) {
        std::snprintf(last_error, sizeof(last_error), "PPO workspace required");
        return 1;
    }
    {
        Timer timer(w, 2);
        size_t bytes =
            static_cast<size_t>(w->models) * w->parameters * sizeof(float);
        if (!download(w, online, w->weights, bytes) ||
            !download(w, moments, w->moments, bytes) ||
            !download(w, variances, w->variances, bytes))
            return 1;
    }
    size_t count = static_cast<size_t>(w->models) * w->parameters;
    return w->profile_error || !finite_output<float>(online, count) ||
           !finite_output<float>(moments, count) ||
           !finite_output<float>(variances, count);
}
}
