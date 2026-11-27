#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Match CPython 3.12/3.13 builtin sum() for float reductions. Only use
   this for reference sum() calls: explicit += loops remain ordered additions.
   Neumaier compensation also preserves CPython's non-finite/signed-zero finish.
   Reference: Python/bltinmodule.c, builtin_sum_impl (CPython 3.12 and 3.13). */
typedef struct {
    double value, correction;
} FloatSum;
static void sum_add(FloatSum *s, double x) {
    double next = s->value + x;
    s->correction += fabs(s->value) >= fabs(x) ? (s->value - next) + x
                                               : (x - next) + s->value;
    s->value = next;
}
static double sum_finish(FloatSum s) {
    return s.correction && isfinite(s.correction) ? s.value + s.correction
                                                  : s.value;
}

/* No fast-math, FMA contraction, reordered reductions, RNG, or global scratch.
   Buffers remain owned by their individual controller. The GIL remains held. */
static int buffer(PyObject *object, Py_buffer *view, char kind, int writable) {
    if (PyObject_GetBuffer(object, view,
                           PyBUF_FORMAT | PyBUF_C_CONTIGUOUS |
                               (writable ? PyBUF_WRITABLE : 0)) < 0)
        return 0;
    if (view->ndim != 1 || !view->format ||
        strcmp(view->format, kind == 'f' ? "f" : "d") ||
        view->itemsize != (kind == 'f' ? sizeof(float) : sizeof(double))) {
        PyBuffer_Release(view);
        memset(view, 0, sizeof(*view));
        PyErr_SetString(PyExc_TypeError,
                        "Expected a contiguous native float32/float64 vector");
        return 0;
    }
    return 1;
}
static Py_ssize_t size(Py_buffer *view) { return view->len / view->itemsize; }
static int length(Py_buffer *view, Py_ssize_t expected) {
    if (size(view) != expected) {
        PyErr_SetString(PyExc_ValueError,
                        "Kernel vector dimensions do not match");
        return 0;
    }
    return 1;
}
static void release(Py_buffer *views, int n) {
    for (int i = 0; i < n; i++)
        if (views[i].obj)
            PyBuffer_Release(&views[i]);
}
/* Accept tuple/list observations and temporary vectors, but copy only neurons,
   never full weight matrices. Validate numeric conversion before mutating. */
static double *vector(PyObject *object, Py_ssize_t *n) {
    PyObject *seq = PySequence_Fast(object, "Expected a numeric sequence");
    if (!seq)
        return NULL;
    *n = PySequence_Fast_GET_SIZE(seq);
    double *data = PyMem_Malloc(((*n) ? (*n) : 1) * sizeof(double));
    if (!data) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t i = 0; i < *n; i++) {
        data[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            PyMem_Free(data);
            return NULL;
        }
    }
    Py_DECREF(seq);
    return data;
}
static PyObject *affine(PyObject *self, PyObject *args) {
    PyObject *xobj, *wobj, *bobj;
    int activate;
    Py_ssize_t start, stop;
    if (!PyArg_ParseTuple(args, "OOOpnn", &xobj, &wobj, &bobj, &activate,
                          &start, &stop))
        return NULL;
    Py_buffer views[2] = {{0}};
    PyObject *result = NULL;
    double *x = NULL;
    Py_ssize_t n;
    if (!buffer(wobj, &views[0], 'f', 0) || !buffer(bobj, &views[1], 'f', 0))
        goto done;
    x = vector(xobj, &n);
    if (!x)
        goto done;
    if (n <= 0 || start < 0 || stop < start || stop > size(&views[1]) ||
        size(&views[0]) / n != size(&views[1]) || size(&views[0]) % n) {
        PyErr_SetString(PyExc_ValueError, "Invalid affine dimensions/range");
        goto done;
    }
    result = PyList_New(stop - start);
    if (!result)
        goto done;
    const float *w = views[0].buf, *bias = views[1].buf;
    for (Py_ssize_t j = start; j < stop; j++) {
        FloatSum sum = {0};
        for (Py_ssize_t i = 0; i < n; i++)
            sum_add(&sum, x[i] * (double)w[j * n + i]);
        double value = (double)bias[j] + sum_finish(sum);
        PyObject *item = PyFloat_FromDouble(activate ? tanh(value) : value);
        if (!item) {
            Py_CLEAR(result);
            goto done;
        }
        PyList_SET_ITEM(result, j - start, item);
    }
done:
    PyMem_Free(x);
    release(views, 2);
    return result;
}
static PyObject *backward(PyObject *self, PyObject *args) {
    PyObject *xobj, *dobj, *wobj, *dwobj, *dbobj;
    int propagate;
    if (!PyArg_ParseTuple(args, "OOOOOp", &xobj, &dobj, &wobj, &dwobj, &dbobj,
                          &propagate))
        return NULL;
    Py_buffer views[3] = {{0}};
    double *x = NULL, *delta = NULL, *previous = NULL;
    Py_ssize_t n, m;
    PyObject *result = NULL;
    if (!buffer(wobj, &views[0], 'f', 0) || !buffer(dwobj, &views[1], 'd', 1) ||
        !buffer(dbobj, &views[2], 'd', 1))
        goto done;
    x = vector(xobj, &n);
    delta = vector(dobj, &m);
    if (!x || !delta)
        goto done;
    if (n <= 0 || size(&views[0]) / n != m || size(&views[0]) % n ||
        !length(&views[1], size(&views[0])) || !length(&views[2], m)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "Invalid backward dimensions");
        goto done;
    }
    if (propagate) {
        previous = PyMem_Calloc(n, sizeof(double));
        if (!previous) {
            PyErr_NoMemory();
            goto done;
        }
    }
    const float *w = views[0].buf;
    double *dw = views[1].buf, *db = views[2].buf;
    for (Py_ssize_t j = 0; j < m; j++) {
        double d = delta[j];
        db[j] += d;
        for (Py_ssize_t i = 0; i < n; i++) {
            dw[j * n + i] += d * x[i];
            if (propagate)
                previous[i] += d * (double)w[j * n + i];
        }
    }
    if (propagate) {
        result = PyList_New(n);
        if (!result)
            goto done;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *item = PyFloat_FromDouble(previous[i]);
            if (!item) {
                Py_CLEAR(result);
                goto done;
            }
            PyList_SET_ITEM(result, i, item);
        }
    } else {
        Py_INCREF(Py_None);
        result = Py_None;
    }
done:
    PyMem_Free(x);
    PyMem_Free(delta);
    PyMem_Free(previous);
    release(views, 3);
    return result;
}
/* Exact widening: avoid Python float boxing when CUDA publishes FP32 gradients.
 */
static PyObject *widen_f32(PyObject *self, PyObject *args) {
    PyObject *source, *destination;
    Py_buffer views[2] = {{0}};
    if (!PyArg_ParseTuple(args, "OO", &source, &destination))
        return NULL;
    PyObject *result = NULL;
    if (!buffer(source, &views[0], 'f', 0) ||
        !buffer(destination, &views[1], 'd', 1) ||
        !length(&views[1], size(&views[0])))
        goto done;
    const float *src = views[0].buf;
    double *dst = views[1].buf;
    for (Py_ssize_t i = 0; i < size(&views[0]); i++)
        dst[i] = (double)src[i];
    Py_INCREF(Py_None);
    result = Py_None;
done:
    release(views, 2);
    return result;
}

static PyObject *adam(PyObject *self, PyObject *args) {
    PyObject *rows;
    Py_ssize_t count, updates;
    double clip, rate;
    if (!PyArg_ParseTuple(args, "Onddn", &rows, &count, &clip, &rate, &updates))
        return NULL;
    if (!PyList_Check(rows) || count <= 0 || updates <= 0 || clip < 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid Adam arguments");
        return NULL;
    }
    Py_ssize_t n = PyList_GET_SIZE(rows);
    Py_buffer *views = PyMem_Calloc(n ? n * 4 : 1, sizeof(Py_buffer));
    if (!views)
        return PyErr_NoMemory();
    PyObject *result = NULL;
    for (Py_ssize_t j = 0; j < n; j++) {
        PyObject *row = PyList_GET_ITEM(rows, j);
        if (!PyTuple_Check(row) || PyTuple_GET_SIZE(row) != 4) {
            PyErr_SetString(
                PyExc_ValueError,
                "Expected Adam parameter/moment/variance/gradient rows");
            goto done;
        }
        for (int k = 0; k < 4; k++)
            if (!buffer(PyTuple_GET_ITEM(row, k), &views[j * 4 + k],
                        k == 3 ? 'd' : 'f', k != 3))
                goto done;
        for (int k = 1; k < 4; k++)
            if (!length(&views[j * 4 + k], size(&views[j * 4])))
                goto done;
    }
    FloatSum total = {0};
    for (Py_ssize_t j = 0; j < n; j++) {
        const double *g = views[j * 4 + 3].buf;
        for (Py_ssize_t i = 0; i < size(&views[j * 4 + 3]); i++)
            sum_add(&total, pow(g[i] / (double)count, 2.0));
    }
    double norm = sqrt(sum_finish(total)),
           factor = fmin(1.0, clip / (norm + 1e-12)) / (double)count;
    double correction1 = 1.0 - pow(0.9, (double)updates),
           correction2 = 1.0 - pow(0.999, (double)updates);
    for (Py_ssize_t j = 0; j < n; j++) {
        float *p = views[j * 4].buf, *m = views[j * 4 + 1].buf,
              *v = views[j * 4 + 2].buf;
        const double *gradient = views[j * 4 + 3].buf;
        for (Py_ssize_t i = 0; i < size(&views[j * 4]); i++) {
            double g = gradient[i] * factor;
            m[i] = (float)(0.9 * (double)m[i] + 0.1 * g);
            v[i] = (float)(0.999 * (double)v[i] + 0.001 * g * g);
            p[i] = (float)((double)p[i] -
                           rate * ((double)m[i] / correction1) /
                               (sqrt((double)v[i] / correction2) + 1e-8));
        }
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    release(views, n * 4);
    PyMem_Free(views);
    return result;
}
static PyObject *q_backward(PyObject *self, PyObject *args) {
    PyObject *xobj, *hobj, *wobj, *gw1, *gb1, *gw2, *gb2;
    Py_ssize_t action;
    double delta;
    if (!PyArg_ParseTuple(args, "OOOnOOOOd", &xobj, &hobj, &wobj, &action, &gw1,
                          &gb1, &gw2, &gb2, &delta))
        return NULL;
    Py_buffer views[5] = {{0}};
    double *x = NULL, *h = NULL;
    PyObject *result = NULL;
    Py_ssize_t n, m;
    PyObject *objects[5] = {wobj, gw1, gb1, gw2, gb2};
    for (int k = 0; k < 5; k++)
        if (!buffer(objects[k], &views[k], k == 0 ? 'f' : 'd', k != 0))
            goto done;
    x = vector(xobj, &n);
    h = vector(hobj, &m);
    if (!x || !h)
        goto done;
    if (n <= 0 || m <= 0 || action < 0 || action >= size(&views[4]) ||
        size(&views[1]) / n != m || size(&views[1]) % n ||
        !length(&views[2], m) || size(&views[0]) / m != size(&views[4]) ||
        size(&views[0]) % m || !length(&views[3], size(&views[0]))) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "Invalid Q gradient dimensions");
        goto done;
    }
    const float *w = views[0].buf;
    double *dw1 = views[1].buf, *db1 = views[2].buf, *dw2 = views[3].buf,
           *db2 = views[4].buf;
    db2[action] += delta;
    for (Py_ssize_t j = 0; j < m; j++) {
        dw2[action * m + j] += delta * h[j];
        double dh = delta * (double)w[action * m + j] * (1.0 - pow(h[j], 2.0));
        db1[j] += dh;
        for (Py_ssize_t i = 0; i < n; i++)
            dw1[j * n + i] += dh * x[i];
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    PyMem_Free(x);
    PyMem_Free(h);
    release(views, 5);
    return result;
}
static PyObject *sgd(PyObject *self, PyObject *args) {
    PyObject *pobj, *gobj;
    double scale;
    Py_buffer views[2] = {{0}};
    PyObject *result = NULL;
    if (!PyArg_ParseTuple(args, "OOd", &pobj, &gobj, &scale))
        return NULL;
    if (!buffer(pobj, &views[0], 'f', 1) || !buffer(gobj, &views[1], 'd', 0) ||
        !length(&views[1], size(&views[0])))
        goto done;
    float *p = views[0].buf;
    const double *g = views[1].buf;
    for (Py_ssize_t i = 0; i < size(&views[0]); i++)
        p[i] = (float)((double)p[i] - scale * g[i]);
    Py_INCREF(Py_None);
    result = Py_None;
done:
    release(views, 2);
    return result;
}
#include "_kernels_data.h"
#include "_kernels_math.h"
#include "_kernels_spatial.h"

static PyMethodDef methods[] = {
    {"groups", groups_native, METH_VARARGS,
     "Sparse native spatial group statistics."},
    {"varint", varint_native, METH_O, "Encode a bounded unsigned integer."},
    {"integer_array", integer_array_native, METH_O,
     "Encode signed integer arrays."},
    {"decode_integers", decode_ints_native, METH_VARARGS,
     "Decode signed integer arrays."},
    {"clone_tree", clone_native, METH_O, "Copy primitive replay trees."},
    {"same_tree", same_native, METH_VARARGS,
     "Type/signed-zero exact replay comparison."},
    {"resource_growth", grow_native, METH_VARARGS,
     "Batch resource growth, original numeric dispatch."},
    {"flat_metabolism", metabolism_native, METH_VARARGS,
     "Batch flat-body costs/age/starvation."},
    {"ray", ray_native, METH_VARARGS, "Fused visibility and water ray."},
    {"nearest_many", nearest_many_native, METH_VARARGS,
     "Ordered native top-k visible targets."},
    {"nearest", nearest_native, METH_VARARGS,
     "Ordered native nearest visible target."},
    {"group_pairs", group_pairs_native, METH_VARARGS,
     "Union-find of nearby animals."},
    {"distribution", policy_distribution_native, METH_VARARGS,
     "Ordered softmax/log probabilities."},
    {"ppo_loss", ppo_loss_native, METH_VARARGS, "PPO objective and gradient."},
    {"quantile", quantile_native, METH_VARARGS,
     "Pairwise quantile Huber objective."},
    {"lstm_step", lstm_step_native, METH_VARARGS,
     "LSTM four gates and hidden/cell update."},
    {"lstm_backward", lstm_backward_native, METH_VARARGS,
     "LSTM gate gradients and hidden/cell BPTT."},
    {"gru_step", gru_step_native, METH_VARARGS, "GRU gates and hidden update."},
    {"gru_backward", gru_backward_native, METH_VARARGS,
     "GRU BPTT accumulation."},
    {"mlp_predict_many", mlp_predict_many_native, METH_O,
     "Independent dense models with shared query-local scratch."},
    {"mlp_predict", mlp_predict_native, METH_VARARGS, "Fused dense inference."},
    {"masked_ppo_batch", masked_ppo_batch_native, METH_VARARGS,
     "Exact masked PPO minibatch."},
    {"dqn_batch", dqn_batch_native, METH_VARARGS,
     "Scalar DQN/Double DQN minibatch gradients."},
    {"auxiliary_batch", auxiliary_batch_native, METH_VARARGS,
     "Exact private consequence-head minibatch gradients."},
    {"late_ppo_batch", late_ppo_batch_native, METH_VARARGS,
     "Exact PPO actor and private late-adapter minibatch gradients."},
    {"ppo_batch", ppo_batch_native, METH_VARARGS,
     "Fused PPO batch gradients and metrics."},
    {"affine", affine, METH_VARARGS,
     "Dense layer, optional tanh, and output range."},
    {"backward", backward, METH_VARARGS,
     "Accumulate dense gradients in float64 buffers."},
    {"adam", adam, METH_VARARGS, "Clipped Adam with float32 storage rounding."},
    {"widen_f32", widen_f32, METH_VARARGS,
     "Exactly widen packed float32 into float64."},
    {"q_backward", q_backward, METH_VARARGS, "Chosen-action Q gradients."},
    {"sgd", sgd, METH_VARARGS,
     "Float32 parameter update with double arithmetic."},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_kernels", NULL, -1,
                                    methods};
PyMODINIT_FUNC PyInit__kernels(void) {
    PyObject *m = PyModule_Create(&module);
    if (!m)
        return NULL;
    if (PyModule_AddIntConstant(m, "api_version", 2) < 0 ||
        PyModule_AddIntConstant(m, "escape_movement", 1) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
