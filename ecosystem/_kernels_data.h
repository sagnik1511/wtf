/* Lossless helpers keep existing record tags, integer widths and object types. */
#include <stdint.h>
static PyObject *varint_native(PyObject *self, PyObject *o) {
    unsigned long long n = PyLong_AsUnsignedLongLong(o);
    if (PyErr_Occurred())
        return NULL;
    unsigned char bytes[10];
    Py_ssize_t k = 0;
    while (n >= 128) {
        bytes[k++] = (unsigned char)((n & 127) | 128);
        n >>= 7;
    }
    bytes[k++] = (unsigned char)n;
    return PyByteArray_FromStringAndSize((char *)bytes, k);
}
static PyObject *integer_array_native(PyObject *self, PyObject *o) {
    PyObject *seq = PySequence_Fast(o, "Expected integer vector");
    if (!seq)
        return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n > PY_SSIZE_T_MAX / 8) {
        Py_DECREF(seq);
        return PyErr_NoMemory();
    }
    PyObject *r = PyByteArray_FromStringAndSize(NULL, n * 8);
    if (!r) {
        Py_DECREF(seq);
        return NULL;
    }
    unsigned char *dest = (unsigned char *)PyByteArray_AS_STRING(r);
    Py_ssize_t count = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        long long value = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) {
            Py_CLEAR(r);
            break;
        }
        if (value > 9007199254740991LL || value < -9007199254740991LL) {
            PyErr_SetString(PyExc_ValueError, "Integer outside exact range");
            Py_CLEAR(r);
            break;
        }
        unsigned long long v = value >= 0 ? (unsigned long long)value * 2
                                          : (unsigned long long)(-value) * 2 - 1;
        while (v >= 128) {
            dest[count++] = (unsigned char)((v & 127) | 128);
            v >>= 7;
        }
        dest[count++] = (unsigned char)v;
    }
    Py_DECREF(seq);
    if (r && PyByteArray_Resize(r, count) < 0)
        Py_CLEAR(r);
    return r;
}
static PyObject *decode_ints_native(PyObject *self, PyObject *args) {
    PyObject *o;
    Py_ssize_t start, count;
    if (!PyArg_ParseTuple(args, "Onn", &o, &start, &count))
        return NULL;
    Py_buffer v = {0};
    if (PyObject_GetBuffer(o, &v, PyBUF_SIMPLE) < 0)
        return NULL;
    PyObject *result = NULL, *a = NULL;
    if (start < 0 || start > v.len || count < 0 || count > v.len - start) {
        PyErr_SetString(PyExc_ValueError, "Truncated binary integer array");
        goto done;
    }
    a = PyList_New(count);
    if (!a)
        goto done;
    const unsigned char *bytes = v.buf;
    Py_ssize_t position = start;
    for (Py_ssize_t i = 0; i < count; i++) {
        unsigned long long value = 0;
        int complete = 0;
        for (int shift = 0; shift < 56; shift += 7) {
            if (position >= v.len) {
                PyErr_SetString(PyExc_ValueError, "Truncated binary value");
                goto done;
            }
            unsigned char b = bytes[position++];
            value += (unsigned long long)(b & 127) << shift;
            if (!(b & 128)) {
                complete = 1;
                break;
            }
        }
        if (!complete || value > 18014398509481983ULL) {
            PyErr_SetString(PyExc_ValueError, "Invalid binary varint");
            goto done;
        }
        long long integer =
            value % 2 ? -(long long)(value / 2) - 1 : (long long)(value / 2);
        PyObject *item = PyLong_FromLongLong(integer);
        if (!item)
            goto done;
        PyList_SET_ITEM(a, i, item);
    }
    result = Py_BuildValue("On", a, position);
done:
    Py_XDECREF(a);
    PyBuffer_Release(&v);
    return result;
}
static PyObject *clone_tree(PyObject *o, PyObject *memo, int depth) {
    if (o == Py_None || PyBool_Check(o) || PyLong_CheckExact(o) ||
        PyFloat_CheckExact(o) || PyUnicode_CheckExact(o)) {
        Py_INCREF(o);
        return o;
    }
    if (depth > 128 || (!PyDict_CheckExact(o) && !PyList_CheckExact(o))) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    PyObject *key = PyLong_FromVoidPtr(o);
    if (!key)
        return NULL;
    PyObject *found = PyDict_GetItemWithError(memo, key);
    if (found) {
        Py_INCREF(found);
        Py_DECREF(key);
        return found;
    }
    PyObject *copy =
        PyDict_CheckExact(o) ? PyDict_New() : PyList_New(PyList_GET_SIZE(o));
    if (!copy) {
        Py_DECREF(key);
        return NULL;
    }
    if (PyDict_SetItem(memo, key, copy) < 0) {
        Py_DECREF(key);
        Py_DECREF(copy);
        return NULL;
    }
    Py_DECREF(key);
    if (PyList_CheckExact(o)) {
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(o); i++) {
            PyObject *v = clone_tree(PyList_GET_ITEM(o, i), memo, depth + 1);
            if (!v) {
                Py_DECREF(copy);
                return NULL;
            }
            if (v == Py_NotImplemented) {
                Py_DECREF(copy);
                return v;
            }
            PyList_SET_ITEM(copy, i, v);
        }
    } else {
        Py_ssize_t pos = 0;
        PyObject *k, *v;
        while (PyDict_Next(o, &pos, &k, &v)) {
            if (!(PyUnicode_CheckExact(k) || PyLong_CheckExact(k) ||
                  PyFloat_CheckExact(k) || PyBool_Check(k) || k == Py_None)) {
                Py_DECREF(copy);
                Py_INCREF(Py_NotImplemented);
                return Py_NotImplemented;
            }
            PyObject *item = clone_tree(v, memo, depth + 1);
            if (!item) {
                Py_DECREF(copy);
                return NULL;
            }
            if (item == Py_NotImplemented) {
                Py_DECREF(copy);
                return item;
            }
            int result = PyDict_SetItem(copy, k, item);
            Py_DECREF(item);
            if (result < 0) {
                Py_DECREF(copy);
                return NULL;
            }
        }
    }
    return copy;
}
static PyObject *clone_native(PyObject *self, PyObject *o) {
    PyObject *memo = PyDict_New();
    if (!memo)
        return NULL;
    PyObject *r = clone_tree(o, memo, 0);
    Py_DECREF(memo);
    return r;
}
static int same_tree(PyObject *a, PyObject *b, int depth) {
    if (Py_TYPE(a) != Py_TYPE(b))
        return 0;
    if (depth > 128)
        return 2;
    if (PyFloat_CheckExact(a)) {
        double x = PyFloat_AS_DOUBLE(a), y = PyFloat_AS_DOUBLE(b);
        return x == y && (x != 0.0 || copysign(1.0, x) == copysign(1.0, y));
    }
    if (PyDict_CheckExact(a)) {
        if (PyDict_Size(a) != PyDict_Size(b))
            return 0;
        Py_ssize_t pos = 0;
        PyObject *k, *v;
        while (PyDict_Next(a, &pos, &k, &v)) {
            PyObject *other = PyDict_GetItemWithError(b, k);
            if (!other)
                return PyErr_Occurred() ? -1 : 0;
            int r = same_tree(v, other, depth + 1);
            if (r != 1)
                return r;
        }
        return 1;
    }
    if (PyList_CheckExact(a) || PyTuple_CheckExact(a)) {
        Py_ssize_t n = PySequence_Size(a);
        if (n != PySequence_Size(b))
            return 0;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *x = PyList_CheckExact(a) ? PyList_GET_ITEM(a, i)
                                               : PyTuple_GET_ITEM(a, i),
                     *y = PyList_CheckExact(b) ? PyList_GET_ITEM(b, i)
                                               : PyTuple_GET_ITEM(b, i);
            int r = same_tree(x, y, depth + 1);
            if (r != 1)
                return r;
        }
        return 1;
    }
    if (PyLong_CheckExact(a) || PyBool_Check(a) || PyUnicode_CheckExact(a) ||
        a == Py_None)
        return PyObject_RichCompareBool(a, b, Py_EQ);
    return 2;
}
static PyObject *same_native(PyObject *self, PyObject *args) {
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b))
        return NULL;
    int r = same_tree(a, b, 0);
    if (r < 0)
        return NULL;
    if (r == 2) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    return PyBool_FromLong(r);
}
/* Resource and flat-body phases retain Python numeric dispatch so int/float
   representation (including min/max ties) is identical to the reference. */
static PyObject *grow_native(PyObject *self, PyObject *args) {
    PyObject *items, *cls, *cap, *growth;
    if (!PyArg_ParseTuple(args, "OOOO", &items, &cls, &cap, &growth))
        return NULL;
    PyObject *seq = PySequence_Fast(items, "Expected resources");
    if (!seq)
        return NULL;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++)
        if ((PyObject *)Py_TYPE(PySequence_Fast_GET_ITEM(seq, i)) != cls) {
            Py_DECREF(seq);
            Py_RETURN_FALSE;
        }
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) {
        PyObject *dict =
            PyObject_GenericGetDict(PySequence_Fast_GET_ITEM(seq, i), NULL);
        if (!dict) {
            Py_DECREF(seq);
            return NULL;
        }
        PyObject *q = PyDict_GetItemString(dict, "quantity");
        if (!q) {
            Py_DECREF(dict);
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "Missing quantity");
            return NULL;
        }
        PyObject *sum = PyNumber_Add(q, growth);
        if (!sum) {
            Py_DECREF(dict);
            Py_DECREF(seq);
            return NULL;
        }
        int less = PyObject_RichCompareBool(sum, cap, Py_LT);
        if (less < 0 || PyDict_SetItemString(dict, "quantity", less ? sum : cap) < 0) {
            Py_DECREF(sum);
            Py_DECREF(dict);
            Py_DECREF(seq);
            return NULL;
        }
        Py_DECREF(sum);
        Py_DECREF(dict);
    }
    Py_DECREF(seq);
    Py_RETURN_TRUE;
}
static PyObject *metabolism_native(PyObject *self, PyObject *args) {
    PyObject *items, *actions, *species, *rewards, *events, *eventcls, *animalcls;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &items, &actions, &species, &rewards,
                          &events, &eventcls, &animalcls))
        return NULL;
    PyObject *seq = PySequence_Fast(items, "Expected bodies");
    if (!seq)
        return NULL;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++)
        if ((PyObject *)Py_TYPE(PySequence_Fast_GET_ITEM(seq, i)) != animalcls) {
            Py_DECREF(seq);
            Py_RETURN_FALSE;
        }
    PyObject *zero = PyFloat_FromDouble(0.0), *one = PyLong_FromLong(1),
             *minusone = PyLong_FromLong(-1),
             *kind = PyUnicode_FromString("damage_taken"),
             *cause = PyUnicode_FromString("starvation"),
             *default_zero = PyLong_FromLong(0), *result = NULL;
    if (!zero || !one || !minusone || !kind || !cause || !default_zero)
        goto done;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) {
        PyObject *dict =
                     PyObject_GenericGetDict(PySequence_Fast_GET_ITEM(seq, i), NULL),
                 *cost = NULL, *remaining = NULL, *term = NULL, *reward = NULL,
                 *age = NULL, *healthnew = NULL, *event = NULL;
        if (!dict)
            goto done;
        PyObject *id = PyDict_GetItemString(dict, "id"),
                 *sp = PyDict_GetItemString(dict, "species"),
                 *energy = PyDict_GetItemString(dict, "energy"),
                 *health = PyDict_GetItemString(dict, "health"),
                 *oldage = PyDict_GetItemString(dict, "age");
        if (!id || !sp || !energy || !health || !oldage) {
            PyErr_SetString(PyExc_ValueError, "Invalid animal");
            goto body_fail;
        }
        PyObject *spec = PyDict_GetItemWithError(species, sp),
                 *action = PyDict_GetItemWithError(actions, id),
                 *before = PyDict_GetItemWithError(rewards, id);
        if (!spec || !action || !before) {
            PyErr_SetString(PyExc_ValueError,
                            "Missing body configuration/action/reward");
            goto body_fail;
        }
        PyObject *body = PyDict_GetItemString(spec, "body"),
                 *r = PyDict_GetItemString(spec, "rewards");
        if (!body || !r) {
            PyErr_SetString(PyExc_ValueError, "Invalid species body");
            goto body_fail;
        }
        cost = PyDict_GetItemString(body, "living_cost");
        if (!cost) {
            PyErr_SetString(PyExc_ValueError, "Missing living cost");
            goto body_fail;
        }
        Py_INCREF(cost);
        if (!PyUnicode_Check(action)) {
            PyErr_SetString(PyExc_TypeError, "Invalid action");
            goto body_fail;
        }
        const char *name = PyUnicode_AsUTF8(action);
        if (!name)
            goto body_fail;
        const char *extra = (!strcmp(name, "move_forward") || !strcmp(name, "swim"))
                                ? "movement_cost"
                            : !strcmp(name, "attack") ? "attack_cost"
                                                      : NULL;
        if (extra) {
            PyObject *c = PyDict_GetItemString(body, extra);
            if (!c) {
                PyErr_SetString(PyExc_ValueError, "Missing action cost");
                goto body_fail;
            }
            PyObject *next = PyNumber_Add(cost, c);
            Py_SETREF(cost, next);
            if (!cost)
                goto body_fail;
        }
        remaining = PyNumber_Subtract(energy, cost);
        if (!remaining)
            goto body_fail;
        int positive = PyObject_RichCompareBool(remaining, zero, Py_GT);
        if (positive < 0)
            goto body_fail;
        if (PyDict_SetItemString(dict, "energy", positive ? remaining : zero) < 0)
            goto body_fail;
        PyObject *coefficient = PyDict_GetItemString(r, "per_energy_cost");
        if (!coefficient)
            coefficient = default_zero;
        term = PyNumber_Multiply(cost, coefficient);
        if (!term)
            goto body_fail;
        reward = PyNumber_Add(before, term);
        if (!reward || PyDict_SetItem(rewards, id, reward) < 0)
            goto body_fail;
        age = PyNumber_Add(oldage, one);
        if (!age || PyDict_SetItemString(dict, "age", age) < 0)
            goto body_fail;
        if (!positive) {
            int alive = PyObject_RichCompareBool(health, zero, Py_GT);
            if (alive < 0)
                goto body_fail;
            if (alive) {
                PyObject *damage = PyDict_GetItemString(body, "starvation_damage");
                if (!damage) {
                    PyErr_SetString(PyExc_ValueError, "Missing starvation damage");
                    goto body_fail;
                }
                int smaller = PyObject_RichCompareBool(damage, health, Py_LT);
                if (smaller < 0)
                    goto body_fail;
                if (!smaller)
                    damage = health;
                Py_INCREF(damage);
                healthnew = PyNumber_Subtract(health, damage);
                if (!healthnew || PyDict_SetItemString(dict, "health", healthnew) < 0) {
                    Py_DECREF(damage);
                    goto body_fail;
                }
                event = PyObject_CallFunctionObjArgs(eventcls, kind, id, minusone,
                                                     damage, cause, NULL);
                Py_DECREF(damage);
                if (!event || PyList_Append(events, event) < 0)
                    goto body_fail;
            }
        }
        Py_DECREF(dict);
        Py_XDECREF(cost);
        Py_XDECREF(remaining);
        Py_XDECREF(term);
        Py_XDECREF(reward);
        Py_XDECREF(age);
        Py_XDECREF(healthnew);
        Py_XDECREF(event);
        continue;
    body_fail:
        Py_DECREF(dict);
        Py_XDECREF(cost);
        Py_XDECREF(remaining);
        Py_XDECREF(term);
        Py_XDECREF(reward);
        Py_XDECREF(age);
        Py_XDECREF(healthnew);
        Py_XDECREF(event);
        goto done;
    }
    result = Py_True;
    Py_INCREF(result);
done:
    Py_DECREF(seq);
    Py_XDECREF(zero);
    Py_XDECREF(one);
    Py_XDECREF(minusone);
    Py_XDECREF(kind);
    Py_XDECREF(cause);
    Py_XDECREF(default_zero);
    return result;
}
