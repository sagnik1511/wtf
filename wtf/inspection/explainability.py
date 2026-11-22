"""Read-only archived-weight inference and local sensitivity; no live controllers."""

from wtf.simulation.tracking import tracking_names

from array import array
import math

MAX_PARAMETERS = 1_000_000
MAX_UNITS = 1024
MAX_LAYERS = 10
VISIBLE_UNITS = 12
POLICIES = {"ppo", "a2c", "reinforce", "grpo", "recurrent_ppo", "maskable_ppo", "mappo"}


def feature_names(signature):
    names = [
        "Energy",
        "Health",
        "Heading x",
        "Heading y",
        "Blocked front",
        "Blocked right",
        "Blocked back",
        "Blocked left",
    ]
    for category in ("Food", "Prey", "Predator"):
        for slot in range(1, signature.get("observation_top_k", 1) + 1):
            names.extend(
                f"{category} {slot} {part}"
                for part in ("dx", "dy", "distance", "present")
            )
    version = signature.get("observation_version", 1)
    if version >= 2:
        names.extend(["Water depth", "Flow x", "Flow y", "Elevation"])
        for side in ("Front", "Right", "Back", "Left"):
            names.extend([f"{side} water", f"{side} climb"])
    if version >= 3:
        names.extend(
            [
                "Sex",
                "Maturity",
                "Cooldown",
                "Self ready",
                "Mate dx",
                "Mate dy",
                "Mate distance",
                "Mate present",
                "Mate ready",
                "Pair opportunity",
            ]
        )
    if version >= 4:
        names.extend(["Self consent", "Mate consent", "Consenting pair"])
    return names + tracking_names(signature)


def prepare_model(model):
    """Validate once and retain bounded packed read-only arrays, not optimizer state."""
    sig = model["signature"]
    kind, inputs, actions = sig["algorithm"], sig["inputs"], sig["actions"]
    if kind not in POLICIES | {"q_network", "qrdqn"}:
        raise ValueError("Unsupported explanation algorithm")
    if (
        type(inputs) is not int
        or not 1 <= inputs <= 4096
        or not 1 <= len(actions) <= 32
        or len(set(actions)) != len(actions)
    ):
        raise ValueError("Invalid explanation input/action dimensions")
    if (
        sig.get("observation_version", 1) not in (1, 2, 3, 4)
        or type(sig.get("observation_top_k", 1)) is not int
        or not 1 <= sig.get("observation_top_k", 1) <= 16
    ):
        raise ValueError("Unsupported explanation observation layout")
    history = sig.get("movement_history_steps", 0)
    scent = sig.get("scent_observation")
    if type(history) is not int or not 0 <= history <= 16:
        raise ValueError("Unsupported explanation movement history")
    if scent is not None:
        if not isinstance(scent, dict) or scent.get("mode", "separated") not in (
            "separated",
            "blended",
        ):
            raise ValueError("Unsupported explanation scent layout")
        if (
            type(scent.get("radius", 2)) is not int
            or not 1 <= scent.get("radius", 2) <= 8
        ):
            raise ValueError("Unsupported explanation scent layout")
        if scent.get("mode") == "blended":
            if set(scent) - {"mode", "radius"}:
                raise ValueError("Unsupported explanation scent layout")
        else:
            names = scent.get("species")
            if (
                not isinstance(names, list)
                or not 1 <= len(names) <= 8
                or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names)
                or type(scent.get("separate_sexes", True)) is not bool
            ):
                raise ValueError("Unsupported explanation scent layout")
    if len(feature_names(sig)) != inputs:
        raise ValueError("Unsupported explanation observation layout")
    count = 0

    def numbers(values, length):
        nonlocal count
        count += length
        if (
            count > MAX_PARAMETERS
            or not isinstance(values, list)
            or len(values) != length
        ):
            raise ValueError("Invalid or oversized explanation parameters")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("Non-finite explanation parameter")
        return array("d", values)

    def dense(layer, width, outputs, activation):
        if type(outputs) is not int or not 1 <= outputs <= MAX_UNITS:
            raise ValueError("Invalid explanation layer width")
        return dict(
            kind="dense",
            inputs=width,
            size=outputs,
            weights=numbers(layer["weights"], width * outputs),
            bias=numbers(layer["bias"], outputs),
            activation=activation,
        )

    layers = []
    p = model["parameters"]
    if kind == "q_network":
        hidden = sig["hidden_layers"][0]
        layers = [
            dense(dict(weights=p["w1"], bias=p["b1"]), inputs, hidden, "tanh"),
            dense(dict(weights=p["w2"], bias=p["b2"]), hidden, len(actions), "linear"),
        ]
    else:
        p = list(p)
        width = inputs
        if kind == "recurrent_ppo":
            h = sig["recurrent_size"]
            if type(h) is not int or not 1 <= h <= MAX_UNITS:
                raise ValueError("Invalid recurrent explanation width")
            cell = p.pop(0)
            layers.append(
                dict(
                    kind="gru",
                    inputs=width,
                    size=h,
                    input_weights=numbers(cell["input_weights"], width * h * 3),
                    recurrent_weights=numbers(cell["recurrent_weights"], h * h * 3),
                    bias=numbers(cell["bias"], h * 3),
                    activation="gru",
                )
            )
            width = h
        shapes = list(sig["hidden_layers"])
        if len(shapes) > MAX_LAYERS:
            raise ValueError("Too many explanation layers")
        head = (
            len(actions) * sig.get("num_quantiles", 1)
            if kind == "qrdqn"
            else len(actions) + int(sig.get("value_head", False))
        )
        if kind == "mappo":
            head = len(actions)
            # Central critic is independent of the local actor and is not an action route.
            p = p[: len(shapes) + 1]
        if len(p) != len(shapes) + 1:
            raise ValueError("Invalid explanation layer count")
        for i, (layer, outputs) in enumerate(zip(p, shapes + [head])):
            layers.append(
                dense(layer, width, outputs, "tanh" if i < len(shapes) else "linear")
            )
            width = outputs
    return dict(signature=sig, layers=layers, parameters=count)


def _sigmoid(v):
    return 1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v))


def explain_model(model, features, action=None, mask=None):
    """One forward/reverse pass. Input×gradient is local, not causal attribution."""
    sig, layers = model["signature"], model["layers"]
    if len(features) != sig["inputs"] or any(
        type(v) not in (float, int) or not math.isfinite(v) for v in features
    ):
        raise ValueError("Missing, incompatible or non-finite recorded observations")
    actions, kind = sig["actions"], sig["algorithm"]
    if action is not None and action not in actions:
        raise ValueError("Unknown explanation action")
    activations, jacobians, gates = [list(features)], [], None
    for layer in layers:
        previous, width, size = activations[-1], layer["inputs"], layer["size"]
        if layer["kind"] == "gru":
            # Archived champions have no recurrent history: explicitly use h(t-1)=0.
            w, bias = layer["input_weights"], layer["bias"]
            gate = lambda offset, fn: [
                fn(
                    bias[offset + j]
                    + sum(
                        previous[i] * w[(offset + j) * width + i] for i in range(width)
                    )
                )
                for j in range(size)
            ]
            z, r, n = gate(0, _sigmoid), gate(size, _sigmoid), gate(2 * size, math.tanh)
            current = [(1 - z[j]) * n[j] for j in range(size)]
            jac = array(
                "d",
                (
                    (1 - z[j]) * (1 - n[j] ** 2) * w[(2 * size + j) * width + i]
                    - n[j] * z[j] * (1 - z[j]) * w[j * width + i]
                    for j in range(size)
                    for i in range(width)
                ),
            )
            gates = dict(update=z, reset=r, candidate=n, memory=current)
        else:
            w = layer["weights"]
            current = [
                b + sum(previous[i] * w[j * width + i] for i in range(width))
                for j, b in enumerate(layer["bias"])
            ]
            if layer["activation"] == "tanh":
                current = [math.tanh(v) for v in current]
            jac = None
        activations.append(current)
        jacobians.append(jac)
    raw = activations[-1]
    n_actions = len(actions)
    policy = kind in POLICIES
    valid = [
        mask.get(a, True) if mask and kind == "maskable_ppo" else True for a in actions
    ]
    if not any(valid):
        raise ValueError("Recorded action mask has no allowed actions")
    if kind == "qrdqn":
        q = sig["num_quantiles"]
        scores = [sum(raw[i * q : (i + 1) * q]) / q for i in range(n_actions)]
    elif policy:
        maximum = max(v for v, allowed in zip(raw[:n_actions], valid) if allowed)
        exp = [
            math.exp(v - maximum) if allowed else 0.0
            for v, allowed in zip(raw[:n_actions], valid)
        ]
        total = sum(exp)
        scores = [v / total for v in exp]
    else:
        scores = raw[:n_actions]
    target = (
        actions.index(action)
        if action
        else max(range(n_actions), key=scores.__getitem__)
    )
    gradient = [0.0] * len(raw)
    if policy:
        for i in range(n_actions):
            gradient[i] = scores[target] * (int(i == target) - scores[i])
    elif kind == "qrdqn":
        for i in range(target * q, (target + 1) * q):
            gradient[i] = 1 / q
    else:
        gradient[target] = 1.0
    gradients = [None] * len(activations)
    gradients[-1] = gradient
    for index in range(len(layers) - 1, -1, -1):
        layer, current = layers[index], activations[index + 1]
        width, size = layer["inputs"], layer["size"]
        if layer["kind"] == "gru":
            w, local = jacobians[index], gradient
        else:
            w = layer["weights"]
            local = (
                [v * (1 - a * a) for v, a in zip(gradient, current)]
                if layer["activation"] == "tanh"
                else gradient
            )
        gradient = [
            sum(local[j] * w[j * width + i] for j in range(size)) for i in range(width)
        ]
        gradients[index] = gradient
    names = feature_names(sig)
    ranked = sorted(
        range(len(features)), key=lambda i: abs(features[i] * gradient[i]), reverse=True
    )
    feature_rows = [
        dict(
            index=i,
            label=names[i],
            value=features[i],
            gradient=gradient[i],
            attribution=features[i] * gradient[i],
        )
        for i in range(len(features))
    ]

    def unit_label(index, unit):
        if index == 0:
            return names[unit]
        if index == len(layers):
            if kind == "qrdqn":
                return f"{actions[unit // sig['num_quantiles']]} q{unit % sig['num_quantiles'] + 1}"
            return actions[unit] if unit < n_actions else "Value"
        return str(unit)

    graph, edges, selected = [], [], []
    for index, values in enumerate(activations):
        keep = (
            ranked[:VISIBLE_UNITS]
            if index == 0
            else sorted(
                range(len(values)),
                key=lambda j: abs(values[j] * gradients[index][j]),
                reverse=True,
            )[:VISIBLE_UNITS]
        )
        if index == len(activations) - 1 and kind != "qrdqn":
            keep = list(range(len(values)))
        selected.append(keep)
        if index == 0:
            layer_label = "Observation"
        elif layers[index - 1]["kind"] == "gru":
            layer_label = "GRU · zero prior memory"
        elif index == len(layers):
            layer_label = (
                "Output logits"
                if policy
                else ("Output quantiles" if kind == "qrdqn" else "Output Q")
            )
        else:
            layer_label = f"Hidden {index}"
        graph.append(
            dict(
                label=layer_label,
                size=len(values),
                activation="input" if index == 0 else layers[index - 1]["activation"],
                values=values if index else [],
                nodes=[
                    dict(
                        index=j,
                        label=unit_label(index, j),
                        target=index == len(layers)
                        and (
                            j // sig["num_quantiles"] == target
                            if kind == "qrdqn"
                            else j == target
                        ),
                        value=values[j],
                        bias=(
                            layers[index - 1]["bias"][j]
                            if index and layers[index - 1]["kind"] == "dense"
                            else None
                        ),
                        sensitivity=gradients[index][j],
                    )
                    for j in keep
                ],
            )
        )
    for index, layer in enumerate(layers):
        w = jacobians[index] if layer["kind"] == "gru" else layer["weights"]
        width = layer["inputs"]
        for j in selected[index + 1]:
            best = sorted(
                selected[index],
                key=lambda i: abs(activations[index][i] * w[j * width + i]),
                reverse=True,
            )[:3]
            for i in best:
                edges.append(
                    dict(
                        layer=index,
                        source=i,
                        target=j,
                        weight=w[j * width + i],
                        signal=activations[index][i] * w[j * width + i],
                        local=layer["kind"] == "gru",
                    )
                )
    return dict(
        algorithm=kind,
        target=actions[target],
        policy=policy,
        outputs=[
            dict(action=a, score=v, masked=not valid[i])
            for i, (a, v) in enumerate(zip(actions, scores))
        ],
        features=feature_rows,
        top_features=[feature_rows[i] for i in ranked[:12]],
        layers=graph,
        edges=edges,
        gates=gates,
        recurrent_reset=kind == "recurrent_ppo",
        critic_separate=kind == "mappo",
        parameters=model["parameters"],
        method=(
            "input × local gradient of action probability"
            if policy
            else "input × local gradient of expected return"
        ),
        quantiles=sig.get("num_quantiles"),
    )
