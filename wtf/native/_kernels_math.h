/* Ordered double math for policy objectives, quantiles and recurrent cells.
   Included once by _kernels.c; no global workspaces or random state. */
static PyObject *double_list(const double *x, Py_ssize_t n) {
    PyObject *a = PyList_New(n);
    if (!a)
        return NULL;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *v = PyFloat_FromDouble(x[i]);
        if (!v) {
            Py_DECREF(a);
            return NULL;
        }
        PyList_SET_ITEM(a, i, v);
    }
    return a;
}
static int distribution(const double *x, Py_ssize_t n, double *p,
                        double *logs) {
    if (n < 1) {
        PyErr_SetString(PyExc_ValueError, "Empty policy");
        return 0;
    }
    double maximum = x[0];
    FloatSum accumulator = {0};
    for (Py_ssize_t i = 1; i < n; i++)
        if (x[i] > maximum)
            maximum = x[i];
    for (Py_ssize_t i = 0; i < n; i++) {
        p[i] = exp(x[i] - maximum);
        sum_add(&accumulator, p[i]);
    }
    double total = sum_finish(accumulator), normalizer = log(total);
    for (Py_ssize_t i = 0; i < n; i++) {
        logs[i] = (x[i] - maximum) - normalizer;
        p[i] /= total;
    }
    return 1;
}
static PyObject *policy_distribution_native(PyObject *self, PyObject *args) {
    PyObject *o;
    if (!PyArg_ParseTuple(args, "O", &o))
        return NULL;
    Py_ssize_t n;
    double *x = vector(o, &n), *scratch = NULL;
    PyObject *result = NULL, *p = NULL, *l = NULL;
    if (!x)
        return NULL;
    scratch = PyMem_Calloc(n ? n * 2 : 1, sizeof(double));
    if (!scratch) {
        PyErr_NoMemory();
        goto done;
    }
    if (!distribution(x, n, scratch, scratch + n))
        goto done;
    p = double_list(scratch, n);
    l = double_list(scratch + n, n);
    if (p && l)
        result = PyTuple_Pack(2, p, l);
done:
    Py_XDECREF(p);
    Py_XDECREF(l);
    PyMem_Free(x);
    PyMem_Free(scratch);
    return result;
}
/* info order: policy loss, value loss, entropy, approximate KL, clip fraction.
 */
static int ppo_derivative(const double *out, Py_ssize_t n, Py_ssize_t action,
                          double old, double adv, double target, double clip,
                          double ec, double vc, double *gradient, double *info,
                          double *scratch, double *loss) {
    Py_ssize_t actions = n - 1;
    if (action < 0 || action >= actions) {
        PyErr_SetString(PyExc_ValueError, "Invalid policy action");
        return 0;
    }
    double *p = scratch, *logs = scratch + actions;
    if (!distribution(out, actions, p, logs))
        return 0;
    double lr = fmax(-60.0, fmin(60.0, logs[action] - old)), ratio = exp(lr);
    double clipped = fmax(1.0 - clip, fmin(1.0 + clip, ratio));
    int inactive =
        (adv > 0 && ratio > 1.0 + clip) || (adv < 0 && ratio < 1.0 - clip);
    double pl = -fmin(ratio * adv, clipped * adv),
           d = inactive ? 0.0 : -adv * ratio;
    FloatSum sum = {0};
    for (Py_ssize_t j = 0; j < actions; j++)
        sum_add(&sum, p[j] * logs[j]);
    double entropy = -sum_finish(sum);
    for (Py_ssize_t j = 0; j < actions; j++)
        gradient[j] = d * ((j == action ? 1.0 : 0.0) - p[j]) +
                      ec * p[j] * (logs[j] + entropy);
    double error = out[actions] - target, vl = 0.5 * error * error;
    gradient[actions] = vc * error;
    info[0] = pl;
    info[1] = vl;
    info[2] = entropy;
    info[3] = (ratio - 1.0) - lr;
    info[4] = (double)(fabs(ratio - 1.0) > clip);
    *loss = pl + vc * vl - ec * entropy;
    return 1;
}
static PyObject *ppo_metrics(const double *info) {
    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d}", "policy_loss", info[0],
                         "value_loss", info[1], "entropy", info[2],
                         "approximate_kl", info[3], "clip_fraction", info[4]);
}
static PyObject *ppo_loss_native(PyObject *self, PyObject *args) {
    PyObject *o;
    Py_ssize_t action;
    double old, adv, target, clip, ec, vc;
    if (!PyArg_ParseTuple(args, "Ondddddd", &o, &action, &old, &adv, &target,
                          &clip, &ec, &vc))
        return NULL;
    Py_ssize_t n;
    double *x = vector(o, &n), *g = NULL, info[5], loss;
    PyObject *result = NULL, *gradient = NULL, *metrics = NULL;
    if (!x)
        return NULL;
    g = PyMem_Calloc(n ? n * 3 : 1, sizeof(double));
    if (!g) {
        PyErr_NoMemory();
        goto done;
    }
    if (!ppo_derivative(x, n, action, old, adv, target, clip, ec, vc, g, info,
                        g + n, &loss))
        goto done;
    gradient = double_list(g, n);
    metrics = ppo_metrics(info);
    if (gradient && metrics)
        result = Py_BuildValue("dOO", loss, gradient, metrics);
done:
    Py_XDECREF(gradient);
    Py_XDECREF(metrics);
    PyMem_Free(x);
    PyMem_Free(g);
    return result;
}
static PyObject *quantile_native(PyObject *self, PyObject *args) {
    PyObject *a, *b;
    double k;
    if (!PyArg_ParseTuple(args, "OOd", &a, &b, &k))
        return NULL;
    Py_ssize_t n, m;
    double *x = vector(a, &n), *y = vector(b, &m), *g = NULL;
    PyObject *r = NULL, *gradient = NULL;
    if (!x || !y)
        goto done;
    if (n < 1 || m < 1 || k <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid quantile dimensions/kappa");
        goto done;
    }
    g = PyMem_Calloc(n, sizeof(double));
    if (!g) {
        PyErr_NoMemory();
        goto done;
    }
    double loss = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        double tau = ((double)i + 0.5) / (double)n;
        for (Py_ssize_t j = 0; j < m; j++) {
            double d = y[j] - x[i], w = fabs(tau - (double)(d < 0)),
                   v = fabs(d);
            double huber = v <= k ? 0.5 * d * d : k * (v - 0.5 * k);
            loss += w * huber / k;
            g[i] -= w * fmax(-k, fmin(k, d)) / k;
        }
    }
    double scale = (double)n * (double)m;
    for (Py_ssize_t i = 0; i < n; i++)
        g[i] /= scale;
    gradient = double_list(g, n);
    if (gradient)
        r = Py_BuildValue("dO", loss / scale, gradient);
done:
    Py_XDECREF(gradient);
    PyMem_Free(x);
    PyMem_Free(y);
    PyMem_Free(g);
    return r;
}
static double stable_sigmoid(double v) {
    return v >= 0 ? 1.0 / (1.0 + exp(-v)) : exp(v) / (1.0 + exp(v));
}
static PyObject *gru_step_native(PyObject *self, PyObject *args) {
    PyObject *xo, *ho, *wxo, *who, *bo;
    int tape;
    if (!PyArg_ParseTuple(args, "OOOOOp", &xo, &ho, &wxo, &who, &bo, &tape))
        return NULL;
    Py_buffer v[3] = {{0}};
    Py_ssize_t n, h;
    double *x = vector(xo, &n), *old = vector(ho, &h), *s = NULL;
    PyObject *r = NULL, *cur = NULL, *parts = NULL;
    if (!x || !old)
        goto done;
    if (!buffer(wxo, &v[0], 'f', 0) || !buffer(who, &v[1], 'f', 0) ||
        !buffer(bo, &v[2], 'f', 0))
        goto done;
    if (n < 1 || h < 1 || size(&v[2]) != 3 * h || size(&v[0]) / n != 3 * h ||
        size(&v[0]) % n || size(&v[1]) / h != 3 * h || size(&v[1]) % h) {
        PyErr_SetString(PyExc_ValueError, "Invalid GRU dimensions");
        goto done;
    }
    s = PyMem_Calloc(h * 5, sizeof(double));
    if (!s) {
        PyErr_NoMemory();
        goto done;
    }
    double *z = s, *reset = s + h, *candidate = s + 2 * h, *rh = s + 3 * h,
           *current = s + 4 * h;
    const float *wx = v[0].buf, *wh = v[1].buf, *bias = v[2].buf;
    for (int gate = 0; gate < 3; gate++) {
        const double *previous = gate == 2 ? rh : old;
        for (Py_ssize_t j = 0; j < h; j++) {
            Py_ssize_t row = gate * h + j;
            FloatSum sx = {0}, sh = {0};
            for (Py_ssize_t i = 0; i < n; i++)
                sum_add(&sx, x[i] * (double)wx[row * n + i]);
            for (Py_ssize_t i = 0; i < h; i++)
                sum_add(&sh, previous[i] * (double)wh[row * h + i]);
            double value =
                ((double)bias[row] + sum_finish(sx)) + sum_finish(sh);
            s[gate * h + j] = gate == 2 ? tanh(value) : stable_sigmoid(value);
        }
        if (gate == 1)
            for (Py_ssize_t j = 0; j < h; j++)
                rh[j] = reset[j] * old[j];
    }
    for (Py_ssize_t j = 0; j < h; j++)
        current[j] = (1.0 - z[j]) * candidate[j] + z[j] * old[j];
    cur = double_list(current, h);
    if (!cur)
        goto done;
    if (tape) {
        parts = PyTuple_New(4);
        if (!parts)
            goto done;
        for (int k = 0; k < 4; k++) {
            PyObject *part = double_list(s + k * h, h);
            if (!part)
                goto done;
            PyTuple_SET_ITEM(parts, k, part);
        }
    } else {
        parts = Py_None;
        Py_INCREF(parts);
    }
    r = PyTuple_Pack(2, cur, parts);
done:
    Py_XDECREF(cur);
    Py_XDECREF(parts);
    PyMem_Free(x);
    PyMem_Free(old);
    PyMem_Free(s);
    release(v, 3);
    return r;
}
static PyObject *gru_backward_native(PyObject *self, PyObject *args) {
    PyObject *seqs, *who, *go;
    if (!PyArg_ParseTuple(args, "OOO", &seqs, &who, &go))
        return NULL;
    Py_buffer views[4] = {{0}};
    PyObject *r = NULL;
    double *parts[8] = {0}, *s = NULL;
    Py_ssize_t sizes[8] = {0};
    if (!PyTuple_Check(seqs) || PyTuple_GET_SIZE(seqs) != 8 ||
        !PyDict_Check(go)) {
        PyErr_SetString(PyExc_TypeError, "Invalid GRU tape");
        goto done;
    }
    for (int k = 0; k < 8; k++) {
        parts[k] = vector(PyTuple_GET_ITEM(seqs, k), &sizes[k]);
        if (!parts[k])
            goto done;
    }
    Py_ssize_t n = sizes[0], h = sizes[1];
    for (int k = 2; k < 8; k++)
        if (sizes[k] != h) {
            PyErr_SetString(PyExc_ValueError, "Invalid GRU tape dimensions");
            goto done;
        }
    if (n < 1 || h < 1) {
        PyErr_SetString(PyExc_ValueError, "Empty GRU tape");
        goto done;
    }
    const char *keys[3] = {"input_weights", "recurrent_weights", "bias"};
    if (!buffer(who, &views[0], 'f', 0))
        goto done;
    for (int k = 0; k < 3; k++) {
        PyObject *o = PyDict_GetItemString(go, keys[k]);
        if (!o) {
            PyErr_SetString(PyExc_ValueError, "Missing GRU gradient");
            goto done;
        }
        if (!buffer(o, &views[k + 1], 'd', 1))
            goto done;
    }
    if (size(&views[0]) / h != 3 * h || size(&views[0]) % h) {
        PyErr_SetString(PyExc_ValueError, "Invalid recurrent matrix");
        goto done;
    }
    if (!length(&views[1], 3 * h * n) || !length(&views[2], 3 * h * h) ||
        !length(&views[3], 3 * h))
        goto done;
    s = PyMem_Calloc(h * 6, sizeof(double));
    if (!s) {
        PyErr_NoMemory();
        goto done;
    }
    double *dh = s, *dz = s + h, *dn = s + 2 * h, *drh = s + 3 * h,
           *dr = s + 4 * h, *prev = s + 5 * h;
    double *x = parts[0], *old = parts[1], *z = parts[2], *reset = parts[3],
           *candidate = parts[4], *rh = parts[5];
    const float *wh = views[0].buf;
    for (Py_ssize_t i = 0; i < h; i++) {
        dh[i] = parts[6][i] + parts[7][i];
        dz[i] = dh[i] * (old[i] - candidate[i]) * z[i] * (1.0 - z[i]);
        dn[i] = dh[i] * (1.0 - z[i]) * (1.0 - candidate[i] * candidate[i]);
    }
    for (Py_ssize_t i = 0; i < h; i++) {
        FloatSum sum = {0};
        for (Py_ssize_t j = 0; j < h; j++)
            sum_add(&sum, dn[j] * (double)wh[(2 * h + j) * h + i]);
        drh[i] = sum_finish(sum);
        dr[i] = drh[i] * old[i] * reset[i] * (1.0 - reset[i]);
    }
    for (Py_ssize_t i = 0; i < h; i++) {
        FloatSum sum = {0};
        for (Py_ssize_t j = 0; j < h; j++)
            sum_add(&sum, dz[j] * (double)wh[j * h + i] +
                              dr[j] * (double)wh[(h + j) * h + i]);
        prev[i] = (dh[i] * z[i] + drh[i] * reset[i]) + sum_finish(sum);
    }
    double *dwx = views[1].buf, *dwh = views[2].buf, *db = views[3].buf;
    for (int gate = 0; gate < 3; gate++) {
        const double *delta = gate == 0   ? dz
                              : gate == 1 ? dr
                                          : dn,
                     *previous = gate == 2 ? rh : old;
        for (Py_ssize_t j = 0; j < h; j++) {
            Py_ssize_t row = gate * h + j;
            double d = delta[j];
            db[row] += d;
            for (Py_ssize_t i = 0; i < n; i++)
                dwx[row * n + i] += d * x[i];
            for (Py_ssize_t i = 0; i < h; i++)
                dwh[row * h + i] += d * previous[i];
        }
    }
    r = double_list(prev, h);
done:
    for (int k = 0; k < 8; k++)
        PyMem_Free(parts[k]);
    PyMem_Free(s);
    release(views, 4);
    return r;
}
/* Acquire model buffers once per inference or minibatch, with one activation
   workspace shared across all samples in that call. */
typedef struct {
    Py_buffer w, b, dw, db;
    Py_ssize_t n, m, offset;
} DenseRow;
static void dense_release(DenseRow *rows, Py_ssize_t n) {
    if (!rows)
        return;
    for (Py_ssize_t j = 0; j < n; j++) {
        release(&rows[j].w, 1);
        release(&rows[j].b, 1);
        release(&rows[j].dw, 1);
        release(&rows[j].db, 1);
    }
    PyMem_Free(rows);
}
static DenseRow *dense_rows(PyObject *layers, PyObject *grads,
                            Py_ssize_t inputs, Py_ssize_t *count,
                            Py_ssize_t *total, Py_ssize_t *maximum) {
    if (!PyList_Check(layers) || PyList_GET_SIZE(layers) < 1 || inputs < 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid dense model");
        return NULL;
    }
    *count = PyList_GET_SIZE(layers);
    *total = inputs;
    *maximum = inputs;
    if (grads && (!PyList_Check(grads) || PyList_GET_SIZE(grads) != *count)) {
        PyErr_SetString(PyExc_ValueError, "Invalid dense gradients");
        return NULL;
    }
    DenseRow *rows = PyMem_Calloc(*count, sizeof(DenseRow));
    if (!rows) {
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t j = 0; j < *count; j++) {
        PyObject *layer = PyList_GET_ITEM(layers, j);
        if (!PyDict_Check(layer)) {
            PyErr_SetString(PyExc_TypeError, "Invalid dense layer");
            goto fail;
        }
        PyObject *w = PyDict_GetItemString(layer, "weights"),
                 *b = PyDict_GetItemString(layer, "bias");
        if (!w || !b) {
            PyErr_SetString(PyExc_ValueError, "Missing dense parameters");
            goto fail;
        }
        if (!buffer(w, &rows[j].w, 'f', 0) || !buffer(b, &rows[j].b, 'f', 0))
            goto fail;
        rows[j].n = inputs;
        rows[j].m = size(&rows[j].b);
        rows[j].offset = *total - inputs;
        if (rows[j].m < 1 || size(&rows[j].w) / inputs != rows[j].m ||
            size(&rows[j].w) % inputs || *total > PY_SSIZE_T_MAX - rows[j].m) {
            PyErr_SetString(PyExc_ValueError, "Invalid dense dimensions");
            goto fail;
        }
        *total += rows[j].m;
        if (rows[j].m > *maximum)
            *maximum = rows[j].m;
        if (grads) {
            PyObject *g = PyList_GET_ITEM(grads, j);
            if (!PyDict_Check(g)) {
                PyErr_SetString(PyExc_TypeError, "Invalid gradient layer");
                goto fail;
            }
            PyObject *dw = PyDict_GetItemString(g, "weights"),
                     *db = PyDict_GetItemString(g, "bias");
            if (!dw || !db) {
                PyErr_SetString(PyExc_ValueError, "Missing dense gradients");
                goto fail;
            }
            if (!buffer(dw, &rows[j].dw, 'd', 1) ||
                !buffer(db, &rows[j].db, 'd', 1) ||
                !length(&rows[j].dw, size(&rows[j].w)) ||
                !length(&rows[j].db, rows[j].m))
                goto fail;
        }
        inputs = rows[j].m;
    }
    return rows;
fail:
    dense_release(rows, *count);
    return NULL;
}
static int fill_sequence(PyObject *o, double *dest, Py_ssize_t n) {
    PyObject *s = PySequence_Fast(o, "Expected observation");
    if (!s)
        return 0;
    if (PySequence_Fast_GET_SIZE(s) != n) {
        Py_DECREF(s);
        PyErr_SetString(PyExc_ValueError, "Observation shape mismatch");
        return 0;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        dest[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(s, i));
        if (PyErr_Occurred()) {
            Py_DECREF(s);
            return 0;
        }
    }
    Py_DECREF(s);
    return 1;
}
static void dense_forward(DenseRow *rows, Py_ssize_t count, double *tape,
                          Py_ssize_t start, Py_ssize_t stop) {
    for (Py_ssize_t j = 0; j < count; j++) {
        DenseRow *r = rows + j;
        const float *w = r->w.buf, *b = r->b.buf;
        double *x = tape + r->offset, *y = x + r->n;
        Py_ssize_t from = j == count - 1 ? start : 0,
                   to = j == count - 1 ? stop : r->m;
        for (Py_ssize_t k = from; k < to; k++) {
            FloatSum sum = {0};
            for (Py_ssize_t i = 0; i < r->n; i++)
                sum_add(&sum, x[i] * (double)w[k * r->n + i]);
            double v = (double)b[k] + sum_finish(sum);
            y[k] = j == count - 1 ? v : tanh(v);
        }
    }
}
static PyObject *mlp_predict_native(PyObject *self, PyObject *args) {
    PyObject *layers, *o;
    Py_ssize_t start, stop;
    if (!PyArg_ParseTuple(args, "OOnn", &layers, &o, &start, &stop))
        return NULL;
    Py_ssize_t n = PySequence_Size(o), count = 0, total = 0, maximum = 0;
    if (n < 0)
        return NULL;
    DenseRow *rows = dense_rows(layers, NULL, n, &count, &total, &maximum);
    if (!rows)
        return NULL;
    PyObject *result = NULL;
    double *tape = NULL;
    Py_ssize_t m = rows[count - 1].m;
    if (start < 0 || stop < start || stop > m) {
        PyErr_SetString(PyExc_ValueError, "Invalid output range");
        goto done;
    }
    tape = PyMem_Calloc(total, sizeof(double));
    if (!tape) {
        PyErr_NoMemory();
        goto done;
    }
    if (!fill_sequence(o, tape, n))
        goto done;
    dense_forward(rows, count, tape, start, stop);
    result = double_list(tape + total - m + start, stop - start);
done:
    PyMem_Free(tape);
    dense_release(rows, count);
    return result;
}
/* Independent dense models share only one query-local activation workspace.
   Each layer keeps the scalar kernel's ordered reductions and private buffers.
 */
static PyObject *mlp_predict_many_native(PyObject *self, PyObject *jobs) {
    PyObject *sequence = PySequence_Fast(jobs, "Expected dense inference jobs");
    if (!sequence)
        return NULL;
    Py_ssize_t length = PySequence_Fast_GET_SIZE(sequence), capacity = 0;
    PyObject *result = PyList_New(length);
    double *tape = NULL;
    if (!result)
        goto done;
    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject *job = PySequence_Fast_GET_ITEM(sequence, i);
        if (!PyTuple_Check(job) || PyTuple_GET_SIZE(job) != 2) {
            PyErr_SetString(PyExc_ValueError, "Invalid dense inference job");
            Py_CLEAR(result);
            goto done;
        }
        PyObject *layers = PyTuple_GET_ITEM(job, 0),
                 *observation = PyTuple_GET_ITEM(job, 1);
        Py_ssize_t n = PySequence_Size(observation), count = 0, total = 0,
                   maximum = 0;
        if (n < 0) {
            Py_CLEAR(result);
            goto done;
        }
        DenseRow *rows = dense_rows(layers, NULL, n, &count, &total, &maximum);
        if (!rows) {
            Py_CLEAR(result);
            goto done;
        }
        if (total > capacity) {
            if (total > PY_SSIZE_T_MAX / sizeof(double)) {
                PyErr_NoMemory();
                dense_release(rows, count);
                Py_CLEAR(result);
                goto done;
            }
            double *grown = PyMem_Realloc(tape, total * sizeof(double));
            if (!grown) {
                PyErr_NoMemory();
                dense_release(rows, count);
                Py_CLEAR(result);
                goto done;
            }
            tape = grown;
            capacity = total;
        }
        Py_ssize_t m = rows[count - 1].m;
        if (!fill_sequence(observation, tape, n)) {
            dense_release(rows, count);
            Py_CLEAR(result);
            goto done;
        }
        dense_forward(rows, count, tape, 0, m);
        PyObject *outputs = double_list(tape + total - m, m);
        dense_release(rows, count);
        if (!outputs) {
            Py_CLEAR(result);
            goto done;
        }
        PyList_SET_ITEM(result, i, outputs);
    }
done:
    PyMem_Free(tape);
    Py_DECREF(sequence);
    return result;
}
static PyObject *ppo_batch_impl(PyObject *args, int masked) {
    PyObject *layers, *grads, *samples;
    double clip, ec, vc;
    if (!PyArg_ParseTuple(args, "OOOddd", &layers, &grads, &samples, &clip, &ec,
                          &vc))
        return NULL;
    if (!PyList_Check(samples) || !PyList_GET_SIZE(samples)) {
        PyErr_SetString(PyExc_ValueError, "Empty PPO batch");
        return NULL;
    }
    PyObject *first = PyList_GET_ITEM(samples, 0);
    if (!PyTuple_Check(first) || PyTuple_GET_SIZE(first) != (masked ? 6 : 5)) {
        PyErr_SetString(PyExc_ValueError, "Invalid PPO sample");
        return NULL;
    }
    Py_ssize_t n = PySequence_Size(PyTuple_GET_ITEM(first, 0)), count = 0,
               total = 0, maximum = 0;
    if (n < 0)
        return NULL;
    DenseRow *rows = dense_rows(layers, grads, n, &count, &total, &maximum);
    if (!rows)
        return NULL;
    Py_ssize_t m = rows[count - 1].m;
    double *tape = NULL, *scratch = NULL, *delta = NULL, *previous = NULL,
           totals[5] = {0};
    PyObject *result = NULL;
    /* Validate the entire sample shape before writing any gradient. */
    for (Py_ssize_t s = 0; s < PyList_GET_SIZE(samples); s++) {
        PyObject *sample = PyList_GET_ITEM(samples, s);
        if (!PyTuple_Check(sample) ||
            PyTuple_GET_SIZE(sample) != (masked ? 6 : 5) ||
            PySequence_Size(PyTuple_GET_ITEM(sample, 0)) != n) {
            PyErr_SetString(PyExc_ValueError, "Invalid PPO sample shape");
            goto done;
        }
        Py_ssize_t action = PyLong_AsSsize_t(PyTuple_GET_ITEM(sample, 1));
        if (PyErr_Occurred())
            goto done;
        if (action < 0 || action >= m - 1) {
            PyErr_SetString(PyExc_ValueError, "Invalid PPO action");
            goto done;
        }
        if (masked) {
            PyObject *mask = PyTuple_GET_ITEM(sample, 5);
            if (!PyTuple_Check(mask) || PyTuple_GET_SIZE(mask) != m - 1) {
                PyErr_SetString(PyExc_ValueError, "Invalid PPO mask shape");
                goto done;
            }
            int valid = 0;
            for (Py_ssize_t k = 0; k < m - 1; k++) {
                PyObject *flag = PyTuple_GET_ITEM(mask, k);
                if (!PyBool_Check(flag)) {
                    PyErr_SetString(PyExc_ValueError,
                                    "PPO mask must contain booleans");
                    goto done;
                }
                valid |= flag == Py_True;
            }
            if (!valid) {
                PyErr_SetString(PyExc_ValueError, "Empty PPO action mask");
                goto done;
            }
        }
    }
    tape = PyMem_Calloc(total, sizeof(double));
    scratch = PyMem_Calloc(4 * maximum, sizeof(double));
    if (!tape || !scratch) {
        PyErr_NoMemory();
        goto done;
    }
    delta = scratch + 2 * maximum;
    previous = delta + maximum;
    for (Py_ssize_t s = 0; s < PyList_GET_SIZE(samples); s++) {
        PyObject *sample = PyList_GET_ITEM(samples, s);
        if (!fill_sequence(PyTuple_GET_ITEM(sample, 0), tape, n))
            goto done;
        Py_ssize_t action = PyLong_AsSsize_t(PyTuple_GET_ITEM(sample, 1));
        double old = PyFloat_AsDouble(PyTuple_GET_ITEM(sample, 2)),
               adv = PyFloat_AsDouble(PyTuple_GET_ITEM(sample, 3)),
               target = PyFloat_AsDouble(PyTuple_GET_ITEM(sample, 4));
        if (PyErr_Occurred())
            goto done;
        dense_forward(rows, count, tape, 0, m);
        if (masked) {
            PyObject *mask = PyTuple_GET_ITEM(sample, 5);
            for (Py_ssize_t k = 0; k < m - 1; k++)
                if (PyTuple_GET_ITEM(mask, k) != Py_True)
                    tape[total - m + k] = -1e30;
        }
        double info[5], loss;
        if (!ppo_derivative(tape + total - m, m, action, old, adv, target, clip,
                            ec, vc, delta, info, scratch, &loss))
            goto done;
        if (masked) {
            PyObject *mask = PyTuple_GET_ITEM(sample, 5);
            for (Py_ssize_t k = 0; k < m - 1; k++)
                if (PyTuple_GET_ITEM(mask, k) != Py_True)
                    delta[k] = 0.0;
        }
        for (Py_ssize_t j = count; j-- > 0;) {
            DenseRow *r = rows + j;
            double *x = tape + r->offset, *dw = r->dw.buf, *db = r->db.buf;
            const float *w = r->w.buf;
            if (j)
                memset(previous, 0, r->n * sizeof(double));
            for (Py_ssize_t k = 0; k < r->m; k++) {
                double d = delta[k];
                db[k] += d;
                for (Py_ssize_t i = 0; i < r->n; i++) {
                    dw[k * r->n + i] += d * x[i];
                    if (j)
                        previous[i] += d * (double)w[k * r->n + i];
                }
            }
            if (j)
                for (Py_ssize_t i = 0; i < r->n; i++)
                    delta[i] = previous[i] * (1.0 - x[i] * x[i]);
        }
        for (int k = 0; k < 5; k++)
            totals[k] += info[k];
    }
    result = ppo_metrics(totals);
done:
    PyMem_Free(tape);
    PyMem_Free(scratch);
    dense_release(rows, count);
    return result;
}

static PyObject *ppo_batch_native(PyObject *self, PyObject *args) {
    return ppo_batch_impl(args, 0);
}
static PyObject *masked_ppo_batch_native(PyObject *self, PyObject *args) {
    return ppo_batch_impl(args, 1);
}

/* LSTM cell only. The MLP head, chronological BPTT boundaries, sampling and
   optimizer cadence remain in Python. Memory is [hidden, cell]. */
static PyObject *lstm_step_native(PyObject *self, PyObject *args) {
    PyObject *xo, *ho, *wxo, *who, *bo;
    int tape;
    if (!PyArg_ParseTuple(args, "OOOOOp", &xo, &ho, &wxo, &who, &bo, &tape))
        return NULL;
    Py_buffer views[3] = {{0}};
    Py_ssize_t n = 0, memory = 0;
    double *x = vector(xo, &n), *old = vector(ho, &memory), *scratch = NULL;
    PyObject *result = NULL, *state = NULL, *parts = NULL;
    if (!x || !old)
        goto done;
    if (!buffer(wxo, &views[0], 'f', 0) || !buffer(who, &views[1], 'f', 0) ||
        !buffer(bo, &views[2], 'f', 0))
        goto done;
    Py_ssize_t h = memory / 2;
    if (n < 1 || memory % 2 || h < 1 || h > 128 || size(&views[2]) != 4 * h ||
        size(&views[0]) / n != 4 * h || size(&views[0]) % n ||
        size(&views[1]) / h != 4 * h || size(&views[1]) % h) {
        PyErr_SetString(PyExc_ValueError, "Invalid LSTM dimensions");
        goto done;
    }
    scratch = PyMem_Calloc(7 * h, sizeof(double));
    if (!scratch) {
        PyErr_NoMemory();
        goto done;
    }
    double *incoming = scratch, *forget = scratch + h,
           *candidate = scratch + 2 * h, *output = scratch + 3 * h,
           *cell = scratch + 4 * h, *current = scratch + 5 * h;
    const float *wx = views[0].buf, *wh = views[1].buf, *bias = views[2].buf;
    for (int gate = 0; gate < 4; gate++) {
        for (Py_ssize_t j = 0; j < h; j++) {
            Py_ssize_t row = gate * h + j;
            FloatSum sx = {0}, sh = {0};
            for (Py_ssize_t i = 0; i < n; i++)
                sum_add(&sx, x[i] * (double)wx[row * n + i]);
            for (Py_ssize_t i = 0; i < h; i++)
                sum_add(&sh, old[i] * (double)wh[row * h + i]);
            double value =
                ((double)bias[row] + sum_finish(sx)) + sum_finish(sh);
            scratch[gate * h + j] =
                gate == 2 ? tanh(value) : stable_sigmoid(value);
        }
    }
    for (Py_ssize_t j = 0; j < h; j++) {
        cell[j] = forget[j] * old[h + j] + incoming[j] * candidate[j];
        current[j] = output[j] * tanh(cell[j]);
        current[h + j] = cell[j];
    }
    state = double_list(current, 2 * h);
    if (!state)
        goto done;
    if (tape) {
        parts = PyTuple_New(5);
        if (!parts)
            goto done;
        for (int k = 0; k < 5; k++) {
            PyObject *part = double_list(scratch + k * h, h);
            if (!part)
                goto done;
            PyTuple_SET_ITEM(parts, k, part);
        }
    } else {
        parts = Py_None;
        Py_INCREF(parts);
    }
    result = PyTuple_Pack(2, state, parts);
done:
    Py_XDECREF(state);
    Py_XDECREF(parts);
    PyMem_Free(x);
    PyMem_Free(old);
    PyMem_Free(scratch);
    release(views, 3);
    return result;
}

static PyObject *lstm_backward_native(PyObject *self, PyObject *args) {
    PyObject *seqs, *who, *go;
    if (!PyArg_ParseTuple(args, "OOO", &seqs, &who, &go))
        return NULL;
    Py_buffer views[4] = {{0}};
    double *parts[9] = {0}, *scratch = NULL;
    Py_ssize_t sizes[9] = {0};
    PyObject *result = NULL;
    if (!PyTuple_Check(seqs) || PyTuple_GET_SIZE(seqs) != 9 ||
        !PyDict_Check(go)) {
        PyErr_SetString(PyExc_TypeError, "Invalid LSTM tape");
        goto done;
    }
    for (int k = 0; k < 9; k++) {
        parts[k] = vector(PyTuple_GET_ITEM(seqs, k), &sizes[k]);
        if (!parts[k])
            goto done;
    }
    Py_ssize_t n = sizes[0], h = sizes[1] / 2;
    if (n < 1 || sizes[1] % 2 || h < 1 || h > 128 || sizes[8] != 2 * h) {
        PyErr_SetString(PyExc_ValueError, "Invalid LSTM memory dimensions");
        goto done;
    }
    for (int k = 2; k < 8; k++)
        if (sizes[k] != h) {
            PyErr_SetString(PyExc_ValueError, "Invalid LSTM tape dimensions");
            goto done;
        }
    if (!buffer(who, &views[0], 'f', 0))
        goto done;
    const char *keys[3] = {"input_weights", "recurrent_weights", "bias"};
    for (int k = 0; k < 3; k++) {
        PyObject *o = PyDict_GetItemString(go, keys[k]);
        if (!o) {
            PyErr_SetString(PyExc_ValueError, "Missing LSTM gradient");
            goto done;
        }
        if (!buffer(o, &views[k + 1], 'd', 1))
            goto done;
    }
    if (size(&views[0]) / h != 4 * h || size(&views[0]) % h ||
        size(&views[1]) / n != 4 * h || size(&views[1]) % n ||
        size(&views[2]) / h != 4 * h || size(&views[2]) % h ||
        !length(&views[3], 4 * h)) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "Invalid LSTM matrix dimensions");
        goto done;
    }
    scratch = PyMem_Calloc(9 * h, sizeof(double));
    if (!scratch) {
        PyErr_NoMemory();
        goto done;
    }
    double *dh = scratch, *tc = scratch + h, *dc = scratch + 2 * h,
           *delta = scratch + 3 * h, *previous = scratch + 7 * h;
    const double *x = parts[0], *old = parts[1], *incoming = parts[2],
                 *forget = parts[3], *candidate = parts[4], *output = parts[5],
                 *cell = parts[6], *derivative = parts[7], *future = parts[8];
    const float *wh = views[0].buf;
    for (Py_ssize_t j = 0; j < h; j++) {
        dh[j] = derivative[j] + future[j];
        tc[j] = tanh(cell[j]);
        /* Python uses ** 2 here; retain libm pow rather than substituting x*x.
         */
        dc[j] = future[h + j] + dh[j] * output[j] * (1.0 - pow(tc[j], 2.0));
        delta[j] = dc[j] * candidate[j] * incoming[j] * (1.0 - incoming[j]);
        delta[h + j] = dc[j] * old[h + j] * forget[j] * (1.0 - forget[j]);
        delta[2 * h + j] = dc[j] * incoming[j] * (1.0 - pow(candidate[j], 2.0));
        delta[3 * h + j] = dh[j] * tc[j] * output[j] * (1.0 - output[j]);
        previous[h + j] = dc[j] * forget[j];
    }
    for (Py_ssize_t i = 0; i < h; i++) {
        FloatSum sum = {0};
        for (int gate = 0; gate < 4; gate++)
            for (Py_ssize_t j = 0; j < h; j++)
                sum_add(&sum, delta[gate * h + j] *
                                  (double)wh[(gate * h + j) * h + i]);
        previous[i] = sum_finish(sum);
    }
    result = double_list(previous, 2 * h);
    if (!result)
        goto done;
    /* All conversion, dimension checks and output allocation precede mutation.
     */
    double *dwx = views[1].buf, *dwh = views[2].buf, *db = views[3].buf;
    for (int gate = 0; gate < 4; gate++)
        for (Py_ssize_t j = 0; j < h; j++) {
            Py_ssize_t row = gate * h + j;
            double d = delta[row];
            db[row] += d;
            for (Py_ssize_t i = 0; i < n; i++)
                dwx[row * n + i] += d * x[i];
            for (Py_ssize_t i = 0; i < h; i++)
                dwh[row * h + i] += d * old[i];
        }
done:
    for (int k = 0; k < 9; k++)
        PyMem_Free(parts[k]);
    PyMem_Free(scratch);
    release(views, 4);
    return result;
}
