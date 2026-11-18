/* Static byte maps and query-local candidate scans. Dynamic revisions and
   per-observer result caches remain owned/invalidation-managed by Python. */
static void trace_ray(Py_ssize_t x, Py_ssize_t y, Py_ssize_t tx, Py_ssize_t ty,
                      const unsigned char *blocked, const unsigned char *water,
                      Py_ssize_t width, Py_ssize_t height, int *visible,
                      Py_ssize_t *wet) {
    Py_ssize_t dx = labs(tx - x), dy = -labs(ty - y), sx = x < tx ? 1 : -1,
               sy = y < ty ? 1 : -1, error = dx + dy;
    *visible = 1;
    *wet = 0;
    while (x != tx || y != ty) {
        Py_ssize_t e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            x += sx;
        }
        if (e2 <= dx) {
            error += dx;
            y += sy;
        }
        if (x >= 0 && x < width && y >= 0 && y < height) {
            Py_ssize_t i = y * width + x;
            if (blocked[i])
                *visible = 0;
            if ((x != tx || y != ty) && water && water[i])
                (*wet)++;
        }
    }
}
static int maps_valid(PyObject *blocked, PyObject *water, Py_ssize_t width,
                      Py_ssize_t height) {
    if (width < 1 || height < 1 || width > 4096 || height > 4096 ||
        !PyBytes_Check(blocked) || !PyBytes_Check(water) ||
        PyBytes_GET_SIZE(blocked) != width * height ||
        PyBytes_GET_SIZE(water) != width * height) {
        PyErr_SetString(PyExc_ValueError, "Invalid static sight maps");
        return 0;
    }
    return 1;
}
static PyObject *ray_native(PyObject *self, PyObject *args) {
    Py_ssize_t x, y, tx, ty, width, height;
    PyObject *blocked, *water;
    if (!PyArg_ParseTuple(args, "nnnnOOnn", &x, &y, &tx, &ty, &blocked, &water, &width,
                          &height))
        return NULL;
    if (!maps_valid(blocked, water, width, height))
        return NULL;
    /* A finite ray bound also rejects overflow/pathological private-API input. */
    if (x < 0 || x >= width || tx < 0 || tx >= width || y < 0 || y >= height ||
        ty < 0 || ty >= height) {
        PyErr_SetString(PyExc_ValueError, "Ray outside map");
        return NULL;
    }
    int visible;
    Py_ssize_t wet;
    trace_ray(x, y, tx, ty, (unsigned char *)PyBytes_AS_STRING(blocked),
              (unsigned char *)PyBytes_AS_STRING(water), width, height, &visible, &wet);
    return Py_BuildValue("On", visible ? Py_True : Py_False, wet);
}
static PyObject *nearest_native(PyObject *self, PyObject *args) {
    PyObject *points, *id, *blocked, *water;
    Py_ssize_t x, y, radius, width, height, base, bonus;
    if (!PyArg_ParseTuple(args, "OnnOnOOnnnn", &points, &x, &y, &id, &radius, &blocked,
                          &water, &width, &height, &base, &bonus))
        return NULL;
    if (!maps_valid(blocked, water, width, height))
        return NULL;
    if (x < 0 || x >= width || y < 0 || y >= height || radius < 1 || radius > 4096 ||
        base < 0 || bonus < 0 || base > 4096 || bonus > 4096) {
        PyErr_SetString(PyExc_ValueError, "Invalid sight range");
        return NULL;
    }
    PyObject *iterator = PyObject_GetIter(points);
    if (!iterator)
        return NULL;
    PyObject *point, *result = NULL;
    Py_ssize_t bx = 0, by = 0, best = PY_SSIZE_T_MAX, best_order = PY_SSIZE_T_MAX;
    int found = 0;
    while ((point = PyIter_Next(iterator))) {
        if (!PyTuple_Check(point) || PyTuple_GET_SIZE(point) != 4) {
            Py_DECREF(point);
            PyErr_SetString(PyExc_ValueError, "Invalid spatial point");
            goto done;
        }
        int same = PyObject_RichCompareBool(PyTuple_GET_ITEM(point, 1), id, Py_EQ);
        if (same < 0) {
            Py_DECREF(point);
            goto done;
        }
        if (same) {
            Py_DECREF(point);
            continue;
        }
        Py_ssize_t order = PyLong_AsSsize_t(PyTuple_GET_ITEM(point, 0)),
                   tx = PyLong_AsSsize_t(PyTuple_GET_ITEM(point, 2)),
                   ty = PyLong_AsSsize_t(PyTuple_GET_ITEM(point, 3));
        Py_DECREF(point);
        if (PyErr_Occurred())
            goto done;
        if (tx < 0 || tx >= width || ty < 0 || ty >= height) {
            PyErr_SetString(PyExc_ValueError, "Spatial target outside map");
            goto done;
        }
        Py_ssize_t dx = tx - x, dy = ty - y, squared = dx * dx + dy * dy;
        if (squared <= radius * radius &&
            (squared < best || (squared == best && order < best_order))) {
            int visible;
            Py_ssize_t wet;
            trace_ray(x, y, tx, ty, (unsigned char *)PyBytes_AS_STRING(blocked),
                      (unsigned char *)PyBytes_AS_STRING(water), width, height,
                      &visible, &wet);
            if (!visible)
                continue;
            Py_ssize_t allowed = base + (wet < bonus ? wet : bonus);
            if (bonus && squared > base * base && squared > allowed * allowed)
                continue;
            bx = tx;
            by = ty;
            best = squared;
            best_order = order;
            found = 1;
        }
    }
    if (PyErr_Occurred())
        goto done;
    if (found)
        result = Py_BuildValue("(nn)", bx - x, by - y);
    else {
        result = Py_None;
        Py_INCREF(result);
    }
done:
    Py_DECREF(iterator);
    return result;
}
/* Union-find on the same candidate pairs as the Python 3x3 bucket scan.
   Bucket construction remains sparse; no O(map-size) scratch for group metrics. */
static Py_ssize_t root_of(Py_ssize_t *parent, Py_ssize_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}
static PyObject *group_pairs_native(PyObject *self, PyObject *args) {
    Py_ssize_t n;
    PyObject *pairs;
    if (!PyArg_ParseTuple(args, "nO", &n, &pairs))
        return NULL;
    if (n < 0 || n > 1000000) {
        PyErr_SetString(PyExc_ValueError, "Invalid group size");
        return NULL;
    }
    Py_ssize_t *parent = PyMem_Calloc(n ? n : 1, sizeof(Py_ssize_t)),
               *sizes = PyMem_Calloc(n ? n : 1, sizeof(Py_ssize_t));
    PyObject *it = NULL, *pair, *r = NULL;
    if (!parent || !sizes) {
        PyErr_NoMemory();
        goto done;
    }
    for (Py_ssize_t i = 0; i < n; i++)
        parent[i] = i;
    it = PyObject_GetIter(pairs);
    if (!it)
        goto done;
    while ((pair = PyIter_Next(it))) {
        Py_ssize_t i, j;
        if (!PyArg_ParseTuple(pair, "nn", &i, &j)) {
            Py_DECREF(pair);
            goto done;
        }
        Py_DECREF(pair);
        if (i < 0 || i >= n || j < 0 || j >= n) {
            PyErr_SetString(PyExc_ValueError, "Invalid group pair");
            goto done;
        }
        parent[root_of(parent, i)] = root_of(parent, j);
    }
    if (PyErr_Occurred())
        goto done;
    for (Py_ssize_t i = 0; i < n; i++)
        sizes[root_of(parent, i)]++;
    Py_ssize_t clusters = 0, solitary = 0, largest = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        clusters += sizes[i] >= 2;
        solitary += sizes[i] == 1;
        if (sizes[i] > largest)
            largest = sizes[i];
    }
    r = Py_BuildValue("{s:n,s:n,s:n}", "clusters", clusters, "solitary", solitary,
                      "largest", largest);
done:
    Py_XDECREF(it);
    PyMem_Free(parent);
    PyMem_Free(sizes);
    return r;
}
static Py_ssize_t floor_cell(Py_ssize_t x, Py_ssize_t r) {
    return x >= 0 ? x / r : (x - r + 1) / r;
}
static PyObject *groups_native(PyObject *self, PyObject *args) {
    PyObject *items;
    Py_ssize_t radius;
    if (!PyArg_ParseTuple(args, "On", &items, &radius))
        return NULL;
    PyObject *seq = PySequence_Fast(items, "Expected animal records");
    if (!seq)
        return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (radius < 1 || radius > 4096 || n > 1000000) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "Invalid grouping dimensions");
        return NULL;
    }
    Py_ssize_t *s = PyMem_Calloc(n ? n * 4 : 1, sizeof(Py_ssize_t));
    PyObject *buckets = PyDict_New(), *r = NULL, *key = NULL, *index = NULL;
    if (!s || !buckets) {
        PyErr_NoMemory();
        goto done;
    }
    Py_ssize_t *x = s, *y = s + n, *parent = s + 2 * n, *sizes = s + 3 * n;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *a = PySequence_Fast_GET_ITEM(seq, i);
        if (!PyDict_Check(a)) {
            Py_INCREF(Py_NotImplemented);
            r = Py_NotImplemented;
            goto done;
        }
        PyObject *xo = PyDict_GetItemString(a, "x"), *yo = PyDict_GetItemString(a, "y");
        if (!xo || !yo || !PyLong_Check(xo) || !PyLong_Check(yo)) {
            Py_INCREF(Py_NotImplemented);
            r = Py_NotImplemented;
            goto done;
        }
        x[i] = PyLong_AsSsize_t(xo);
        y[i] = PyLong_AsSsize_t(yo);
        if (PyErr_Occurred())
            goto done;
        if (labs(x[i]) > 1000000 || labs(y[i]) > 1000000) {
            Py_INCREF(Py_NotImplemented);
            r = Py_NotImplemented;
            goto done;
        }
        parent[i] = i;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        Py_ssize_t bx = floor_cell(x[i], radius), by = floor_cell(y[i], radius);
        for (Py_ssize_t cx = bx - 1; cx <= bx + 1; cx++)
            for (Py_ssize_t cy = by - 1; cy <= by + 1; cy++) {
                key = Py_BuildValue("nn", cx, cy);
                if (!key)
                    goto done;
                PyObject *bucket = PyDict_GetItemWithError(buckets, key);
                Py_CLEAR(key);
                if (PyErr_Occurred())
                    goto done;
                if (!bucket)
                    continue;
                for (Py_ssize_t k = 0; k < PyList_GET_SIZE(bucket); k++) {
                    Py_ssize_t j = PyLong_AsSsize_t(PyList_GET_ITEM(bucket, k)),
                               dx = x[i] - x[j], dy = y[i] - y[j];
                    if (dx * dx + dy * dy <= radius * radius)
                        parent[root_of(parent, i)] = root_of(parent, j);
                }
            }
        key = Py_BuildValue("nn", bx, by);
        if (!key)
            goto done;
        PyObject *bucket = PyDict_GetItemWithError(buckets, key);
        if (!bucket) {
            bucket = PyList_New(0);
            if (!bucket)
                goto done;
            int ok = PyDict_SetItem(buckets, key, bucket);
            Py_DECREF(bucket);
            if (ok < 0)
                goto done;
            bucket = PyDict_GetItem(buckets, key);
        }
        index = PyLong_FromSsize_t(i);
        if (!index || PyList_Append(bucket, index) < 0)
            goto done;
        Py_CLEAR(index);
        Py_CLEAR(key);
    }
    for (Py_ssize_t i = 0; i < n; i++)
        sizes[root_of(parent, i)]++;
    Py_ssize_t clusters = 0, solitary = 0, largest = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        clusters += sizes[i] >= 2;
        solitary += sizes[i] == 1;
        if (sizes[i] > largest)
            largest = sizes[i];
    }
    r = Py_BuildValue("{s:n,s:n,s:n}", "clusters", clusters, "solitary", solitary,
                      "largest", largest);
done:
    Py_XDECREF(key);
    Py_XDECREF(index);
    Py_XDECREF(buckets);
    PyMem_Free(s);
    Py_DECREF(seq);
    return r;
}
