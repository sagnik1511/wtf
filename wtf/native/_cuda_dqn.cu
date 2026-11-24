// Optional Linux/NVCC DQN gradient batches. No CPython, PyTorch or NumPy ABI.
#include <algorithm>
#include <cmath>
#include <cstdio>
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
    return static_cast<int>(code);
}
struct Workspace {
    int models, batch, layers, inputs, outputs, parameters, activations, width;
    bool training;
    int *sizes = nullptr, *po = nullptr, *ao = nullptr;
    float *weights = nullptr, *targets = nullptr;
    double *observations = nullptr, *next = nullptr, *rewards = nullptr;
    int *actions = nullptr, *terminal = nullptr;
    double *scratch = nullptr, *gradients = nullptr, *losses = nullptr,
           *output = nullptr;
    ~Workspace() {
        cudaFree(sizes);
        cudaFree(po);
        cudaFree(ao);
        cudaFree(weights);
        cudaFree(targets);
        cudaFree(observations);
        cudaFree(next);
        cudaFree(rewards);
        cudaFree(actions);
        cudaFree(terminal);
        cudaFree(scratch);
        cudaFree(gradients);
        cudaFree(losses);
        cudaFree(output);
    }
};
template <typename T> bool allocate(T **pointer, size_t count) {
    return fail(cudaMalloc(reinterpret_cast<void **>(pointer),
                           count * sizeof(T))) == 0;
}
__device__ void forward(const float *weights, const double *input, double *tape,
                        const int *sizes, const int *po, const int *ao,
                        int layers) {
    for (int i = threadIdx.x; i < sizes[0]; i += blockDim.x)
        tape[i] = input[i];
    __syncthreads();
    for (int layer = 0; layer < layers; ++layer) {
        int ni = sizes[layer], no = sizes[layer + 1];
        for (int j = threadIdx.x; j < no; j += blockDim.x) {
            double total = 0.;
            for (int i = 0; i < ni; ++i)
                total += tape[ao[layer] + i] *
                         static_cast<double>(weights[po[layer] + j * ni + i]);
            total += static_cast<double>(weights[po[layer] + ni * no + j]);
            tape[ao[layer + 1] + j] = layer + 1 == layers ? total : tanh(total);
        }
        __syncthreads();
    }
}
__global__ void sample_gradients(const float *weights, const float *targets,
                                 const double *observations, const double *next,
                                 const double *rewards, const int *actions,
                                 const int *terminal, const int *sizes,
                                 const int *po, const int *ao, int layers,
                                 int parameters, int activations, int width,
                                 int batch, double discount, double threshold,
                                 int double_q, double *scratch,
                                 double *gradients, double *losses) {
    int sample = blockIdx.x, model = sample / batch, inputs = sizes[0];
    const float *online = weights + static_cast<size_t>(model) * parameters;
    const float *target_weights =
        targets + static_cast<size_t>(model) * parameters;
    double *tape =
        scratch + static_cast<size_t>(sample) * (2 * activations + 2 * width);
    double *aux = tape + activations, *delta = aux + activations,
           *prior_delta = delta + width;
    double *gradient = gradients + static_cast<size_t>(sample) * parameters;
    __shared__ int best;
    __shared__ double target;
    forward(online, observations + static_cast<size_t>(sample) * inputs, tape,
            sizes, po, ao, layers);
    if (!terminal[sample]) {
        if (double_q) {
            forward(online, next + static_cast<size_t>(sample) * inputs, aux,
                    sizes, po, ao, layers);
            if (threadIdx.x == 0) {
                best = 0;
                for (int j = 1; j < sizes[layers]; ++j)
                    if (aux[ao[layers] + j] > aux[ao[layers] + best])
                        best = j;
            }
            __syncthreads();
        }
        forward(target_weights, next + static_cast<size_t>(sample) * inputs,
                aux, sizes, po, ao, layers);
        if (threadIdx.x == 0) {
            if (!double_q) {
                best = 0;
                for (int j = 1; j < sizes[layers]; ++j)
                    if (aux[ao[layers] + j] > aux[ao[layers] + best])
                        best = j;
            }
            target = rewards[sample] + discount * aux[ao[layers] + best];
        }
    } else if (threadIdx.x == 0)
        target = rewards[sample];
    __syncthreads();
    double error = tape[ao[layers] + actions[sample]] - target;
    double derivative = fmax(-threshold, fmin(threshold, error));
    if (threadIdx.x == 0) {
        double absolute = fabs(error);
        losses[sample] = absolute <= threshold
                             ? 0.5 * error * error
                             : threshold * (absolute - 0.5 * threshold);
    }
    for (int j = threadIdx.x; j < sizes[layers]; j += blockDim.x)
        delta[j] = j == actions[sample] ? derivative : 0.;
    __syncthreads();
    for (int layer = layers - 1; layer >= 0; --layer) {
        int ni = sizes[layer], no = sizes[layer + 1];
        for (int k = threadIdx.x; k < ni * no; k += blockDim.x)
            gradient[po[layer] + k] = delta[k / ni] * tape[ao[layer] + k % ni];
        for (int j = threadIdx.x; j < no; j += blockDim.x)
            gradient[po[layer] + ni * no + j] = delta[j];
        if (layer) {
            for (int i = threadIdx.x; i < ni; i += blockDim.x) {
                double total = 0.;
                for (int j = 0; j < no; ++j)
                    total +=
                        static_cast<double>(online[po[layer] + j * ni + i]) *
                        delta[j];
                double value = tape[ao[layer] + i];
                prior_delta[i] = total * (1. - value * value);
            }
        }
        __syncthreads();
        double *swap = delta;
        delta = prior_delta;
        prior_delta = swap;
    }
}
__global__ void reduce_samples(const double *gradients, const double *losses,
                               double *output, int parameters, int batch,
                               int models) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(models) * (parameters + 1);
    if (index >= count)
        return;
    int model = index / (parameters + 1), parameter = index % (parameters + 1);
    double total = 0.;
    for (int sample = 0; sample < batch; ++sample) {
        size_t row = static_cast<size_t>(model) * batch + sample;
        total += parameter == parameters
                     ? losses[row]
                     : gradients[row * parameters + parameter];
    }
    output[index] = total;
}
} // namespace

extern "C" {
int wtf_cuda_api_version() { return 1; }
const char *wtf_cuda_error() { return last_error; }
int wtf_cuda_probe(char *name, int capacity) {
    int device;
    if (fail(cudaGetDevice(&device)))
        return 1;
    cudaDeviceProp properties;
    if (fail(cudaGetDeviceProperties(&properties, device)))
        return 1;
    std::snprintf(name, capacity, "%s", properties.name);
    return 0;
}
void *wtf_cuda_create(const int *sizes, int layers, int models, int batch,
                      int training) {
    if (layers < 1 || layers > 16 || models < 1 || models > 4096 || batch < 1 ||
        batch > 4096) {
        std::snprintf(last_error, sizeof(last_error),
                      "Invalid CUDA batch dimensions");
        return nullptr;
    }
    if ((training != 0 && training != 1) || (!training && batch != 1)) {
        std::snprintf(last_error, sizeof(last_error),
                      "Invalid CUDA workspace mode");
        return nullptr;
    }
    for (int i = 0; i <= layers; ++i)
        if (sizes[i] < 1 || sizes[i] > 4096) {
            std::snprintf(last_error, sizeof(last_error),
                          "CUDA layer width must be 1..4096");
            return nullptr;
        }
    std::vector<int> po(layers + 1, 0), ao(layers + 2, 0);
    int width = 0;
    for (int i = 0; i <= layers; ++i) {
        width = std::max(width, sizes[i]);
        ao[i + 1] = ao[i] + sizes[i];
        if (i < layers)
            po[i + 1] = po[i] + sizes[i] * sizes[i + 1] + sizes[i + 1];
    }
    auto *w = new (std::nothrow) Workspace;
    if (!w) {
        std::snprintf(last_error, sizeof(last_error), "Host allocation failed");
        return nullptr;
    }
    w->models = models;
    w->batch = batch;
    w->layers = layers;
    w->inputs = sizes[0];
    w->training = training;
    w->outputs = sizes[layers];
    w->parameters = po[layers];
    w->activations = ao[layers + 1];
    w->width = width;
    size_t samples = static_cast<size_t>(models) * batch;
    bool ok =
        allocate(&w->sizes, layers + 1) && allocate(&w->po, layers + 1) &&
        allocate(&w->ao, layers + 2) &&
        allocate(&w->weights, static_cast<size_t>(models) * w->parameters) &&
        allocate(&w->observations, samples * w->inputs) &&
        allocate(&w->scratch, samples * (2 * w->activations + 2 * width)) &&
        allocate(&w->output, static_cast<size_t>(models) *
                                 (training ? w->parameters + 1 : w->outputs));
    if (ok && training)
        ok = allocate(&w->targets,
                      static_cast<size_t>(models) * w->parameters) &&
             allocate(&w->next, samples * w->inputs) &&
             allocate(&w->rewards, samples) && allocate(&w->actions, samples) &&
             allocate(&w->terminal, samples) &&
             allocate(&w->gradients, samples * w->parameters) &&
             allocate(&w->losses, samples);
    if (ok)
        ok = !fail(cudaMemcpy(w->sizes, sizes, (layers + 1) * sizeof(int),
                              cudaMemcpyHostToDevice)) &&
             !fail(cudaMemcpy(w->po, po.data(), (layers + 1) * sizeof(int),
                              cudaMemcpyHostToDevice)) &&
             !fail(cudaMemcpy(w->ao, ao.data(), (layers + 2) * sizeof(int),
                              cudaMemcpyHostToDevice));
    if (!ok) {
        delete w;
        return nullptr;
    }
    return w;
}
void wtf_cuda_destroy(void *pointer) {
    delete static_cast<Workspace *>(pointer);
}
int wtf_cuda_run(void *pointer, const float *weights, const float *targets,
                 const double *observations, const double *next,
                 const double *rewards, const int *actions, const int *terminal,
                 double discount, double threshold, int double_q,
                 double *output) {
    auto *w = static_cast<Workspace *>(pointer);
    if (!w->training) {
        std::snprintf(last_error, sizeof(last_error),
                      "Training workspace required");
        return 1;
    }
    size_t samples = static_cast<size_t>(w->models) * w->batch;
#define COPY(field, host, bytes)                                               \
    if (fail(cudaMemcpy(w->field, host, bytes, cudaMemcpyHostToDevice)))       \
    return 1
    COPY(weights, weights,
         static_cast<size_t>(w->models) * w->parameters * sizeof(float));
    COPY(targets, targets,
         static_cast<size_t>(w->models) * w->parameters * sizeof(float));
    COPY(observations, observations, samples * w->inputs * sizeof(double));
    COPY(next, next, samples * w->inputs * sizeof(double));
    COPY(rewards, rewards, samples * sizeof(double));
    COPY(actions, actions, samples * sizeof(int));
    COPY(terminal, terminal, samples * sizeof(int));
#undef COPY
    sample_gradients<<<samples, 128>>>(
        w->weights, w->targets, w->observations, w->next, w->rewards,
        w->actions, w->terminal, w->sizes, w->po, w->ao, w->layers,
        w->parameters, w->activations, w->width, w->batch, discount, threshold,
        double_q, w->scratch, w->gradients, w->losses);
    if (fail(cudaGetLastError()))
        return 1;
    size_t outputs = static_cast<size_t>(w->models) * (w->parameters + 1);
    reduce_samples<<<(outputs + 255) / 256, 256>>>(
        w->gradients, w->losses, w->output, w->parameters, w->batch, w->models);
    if (fail(cudaGetLastError()))
        return 1;
    return fail(cudaMemcpy(output, w->output, outputs * sizeof(double),
                           cudaMemcpyDeviceToHost));
}
}

namespace {
__global__ void predict_models(const float *weights, const double *observations,
                               double *scratch, double *output,
                               const int *sizes, const int *po, const int *ao,
                               int layers, int parameters, int activations,
                               int width) {
    int model = blockIdx.x;
    double *tape =
        scratch + static_cast<size_t>(model) * (2 * activations + 2 * width);
    forward(weights + static_cast<size_t>(model) * parameters,
            observations + static_cast<size_t>(model) * sizes[0], tape, sizes,
            po, ao, layers);
    for (int j = threadIdx.x; j < sizes[layers]; j += blockDim.x)
        output[static_cast<size_t>(model) * sizes[layers] + j] =
            tape[ao[layers] + j];
}
} // namespace
extern "C" int wtf_cuda_predict(void *pointer, const float *weights,
                                const double *observations, double *output) {
    auto *w = static_cast<Workspace *>(pointer);
    if (w->batch != 1) {
        std::snprintf(last_error, sizeof(last_error),
                      "Prediction workspace must have batch=1");
        return 1;
    }
    if (fail(cudaMemcpy(w->weights, weights,
                        static_cast<size_t>(w->models) * w->parameters *
                            sizeof(float),
                        cudaMemcpyHostToDevice)) ||
        fail(cudaMemcpy(w->observations, observations,
                        static_cast<size_t>(w->models) * w->inputs *
                            sizeof(double),
                        cudaMemcpyHostToDevice)))
        return 1;
    predict_models<<<w->models, 128>>>(
        w->weights, w->observations, w->scratch, w->output, w->sizes, w->po,
        w->ao, w->layers, w->parameters, w->activations, w->width);
    if (fail(cudaGetLastError()))
        return 1;
    return fail(
        cudaMemcpy(output, w->output,
                   static_cast<size_t>(w->models) * w->outputs * sizeof(double),
                   cudaMemcpyDeviceToHost));
}
