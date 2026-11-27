"""Incremental spatial windows; all state is derived and checkpoint independent."""

import math
from itertools import chain
from wtf.native import kernels


class _Bucket:

    # This record/owner deliberately keeps its related fields and lifecycle operations
    # together without artificial wrappers.
    # pylint: disable=too-few-public-methods

    __slots__ = ("points", "version")

    def __init__(self):
        self.points = {}
        self.version = 0


def _cells(bounds):
    left, right, top, bottom = bounds
    return ((x, y) for x in range(left, right + 1) for y in range(top, bottom + 1))


def _difference(bounds, overlap):
    """Visit only strips outside the overlapping rectangle, never its interior."""
    left, right, top, bottom = bounds
    ol, oright, ot, ob = overlap
    middle_left, middle_right = max(left, ol), min(right, oright)
    yield from _cells((left, min(right, ol - 1), top, bottom))
    yield from _cells((max(left, oright + 1), right, top, bottom))
    yield from _cells((middle_left, middle_right, top, min(bottom, ot - 1)))
    yield from _cells((middle_left, middle_right, max(top, ob + 1), bottom))


def _front_points(points, x, y, direction, cosine_squared):
    """Filter once before the existing native range/occlusion/top-k kernel."""
    if direction is None:
        return points
    hx, hy = direction

    def selected():
        for point in points:
            dx, dy = point[2] - x, point[3] - y
            dot = dx * hx + dy * hy
            if dot >= 0 and (
                not cosine_squared or dot * dot >= (dx * dx + dy * dy) * cosine_squared
            ):
                yield point

    return selected()


class SpatialIndex:
    """Bucket world entities for bounded local perception queries."""

    CELL_SIZE = 8

    def __init__(self):
        self.points = {}
        self.buckets = {}
        self.active_cells = set()
        self.windows = {}
        self.results = {}
        self.version = 0

    def sync(self, points, observers, obstacles_changed=False):
        """Update moved/added/removed targets once for all observers.

        Scanning the authoritative entities also detects manual edits between
        steps. Growth above the food-presence threshold needs no invalidation.
        """

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals

        seen, dirty = set(), set()
        size = self.CELL_SIZE
        for order, (animal_id, (x, y)) in enumerate(points):
            key = (x, y) if animal_id is None else animal_id
            seen.add(key)
            point = (order, animal_id, x, y)
            old = self.points.get(key)
            if old == point:
                continue
            if old is not None:
                cell = old[2] // size, old[3] // size
                del self.buckets[cell].points[key]
                dirty.add(cell)
            self.points[key] = point
            cell = x // size, y // size
            if cell not in self.buckets:
                self.buckets[cell] = _Bucket()
            self.buckets[cell].points[key] = point
            dirty.add(cell)
        for key in self.points.keys() - seen:
            old = self.points.pop(key)
            cell = old[2] // size, old[3] // size
            del self.buckets[cell].points[key]
            dirty.add(cell)
        if dirty:
            self.version += 1
            for cell in dirty:
                bucket = self.buckets[cell]
                bucket.version += 1
                if bucket.points:
                    self.active_cells.add(cell)
                else:
                    self.active_cells.discard(cell)
        for animal_id in self.windows.keys() - observers:
            del self.windows[animal_id]
        if obstacles_changed:
            self.results.clear()
        else:
            for animal_id in self.results.keys() - observers:
                del self.results[animal_id]

    def _window(self, animal_id, x, y, radius):

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-locals

        size = self.CELL_SIZE
        bounds = (
            (x - radius) // size,
            (x + radius) // size,
            (y - radius) // size,
            (y + radius) // size,
        )
        left, right, top, bottom = bounds
        if len(self.points) < 32 or (right - left + 1) * (bottom - top + 1) >= len(
            self.active_cells
        ):
            return None  # A direct scan is cheaper for sparse/wide views.
        previous = self.windows.get(animal_id)
        if previous is None:
            cells = {}
            entering = _cells(bounds)
        else:
            old_bounds, cells = previous
            if old_bounds == bounds:
                return cells.values()
            # Reuse overlapping bucket references; touch only leaving/entering
            # strips. This also handles teleporting and changed vision radii.
            for cell in _difference(old_bounds, bounds):
                del cells[cell]
            entering = _difference(bounds, old_bounds)
        for cell in entering:
            if cell not in self.buckets:
                self.buckets[cell] = _Bucket()
            cells[cell] = self.buckets[cell]
        self.windows[animal_id] = bounds, cells
        return cells.values()

    def nearest(self, animal, radius, visible, range_check=None, range_key=None):
        """Find the nearest eligible visible entities with deterministic ordering."""

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-boolean-expressions
        # pylint: disable=too-many-locals
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        ax, ay, animal_id = animal.x, animal.y, animal.id
        buckets = self._window(animal_id, ax, ay, radius)
        if buckets is None:
            revision = self.version
            points = self.points.values()
        else:
            revision = tuple(bucket.version for bucket in buckets)
            points = chain.from_iterable(bucket.points.values() for bucket in buckets)
        key = ax, ay, radius, revision
        if range_check is not None:
            key += (range_key,)
        previous = self.results.get(animal_id)
        if previous is not None and previous[0] == key:
            return previous[1]
        owner = getattr(visible, "__self__", None)
        maps = getattr(owner, "_native_sight", None)
        native_range = getattr(range_check, "_native_sight_range", None)
        if (
            maps is not None
            and kernels.available("nearest")
            and getattr(visible, "__func__", None)
            is getattr(owner, "_native_sight_callback", None)
            and (range_check is None or native_range is not None)
        ):
            base, bonus = native_range if native_range is not None else (radius, 0)
            point = kernels.native.nearest(
                _front_points(
                    points,
                    ax,
                    ay,
                    getattr(range_check, "_native_sight_heading", None),
                    getattr(range_check, "_native_sight_cosine_squared", 0.0),
                ),
                ax,
                ay,
                animal_id,
                radius,
                *maps,
                owner.width,
                owner.height,
                base,
                bonus,
            )
            # CPython 3.9's compensated hypot differs from libm hypot for some
            # integer pairs. Keep the original final feature calculation.
            features = (
                [0.0, 0.0, 0.0, 0.0]
                if point is None
                else [
                    point[0] / radius,
                    point[1] / radius,
                    math.hypot(*point) / radius,
                    1.0,
                ]
            )
            self.results[animal_id] = key, features
            return features
        best, best_squared, best_order = None, float("inf"), float("inf")
        start, radius_squared = (ax, ay), radius * radius
        for order, other_id, x, y in points:
            if other_id == animal_id:
                continue
            dx, dy = x - ax, y - ay
            squared = dx * dx + dy * dy
            if (
                squared <= radius_squared
                and (
                    squared < best_squared
                    or (squared == best_squared and order < best_order)
                )
                and visible(start, (x, y))
                and (range_check is None or range_check(start, (x, y), squared))
            ):
                best, best_squared, best_order = (x, y), squared, order
        features = (
            [0.0, 0.0, 0.0, 0.0]
            if best is None
            else [
                (best[0] - ax) / radius,
                (best[1] - ay) / radius,
                math.hypot(best[0] - ax, best[1] - ay) / radius,
                1.0,
            ]
        )
        self.results[animal_id] = key, features
        return features

    def nearest_many(
        self, animal, radius, visible, k, range_check=None, range_key=None
    ):
        """Distance-ordered visible slots, padded to k; reuse phase-local indexes.

        One result per observer replaces the previous query. Position, target
        bucket revisions, radius, k and directional range settings key it;
        sync clears results on obstacle/visibility edits and removes dead IDs.
        """

        # Keep this ordered phase and its shared state together; splitting it only to
        # meet a count obscures the contract.
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-positional-arguments
        # The optional CPython API is checked by its wrapper and native/reference
        # equivalence tests.
        # pylint: disable=c-extension-no-member

        if k == 1:
            return self.nearest(animal, radius, visible, range_check, range_key)
        ax, ay, animal_id = animal.x, animal.y, animal.id
        buckets = self._window(animal_id, ax, ay, radius)
        if buckets is None:
            revision = self.version
            points = self.points.values()
        else:
            revision = tuple(bucket.version for bucket in buckets)
            points = chain.from_iterable(bucket.points.values() for bucket in buckets)
        key = ax, ay, radius, revision, "top_k", k, range_key
        previous = self.results.get(animal_id)
        if previous is not None and previous[0] == key:
            return previous[1]
        owner = getattr(visible, "__self__", None)
        maps = getattr(owner, "_native_sight", None)
        native_range = getattr(range_check, "_native_sight_range", None)
        if (
            maps is not None
            and kernels.available("nearest_many")
            and getattr(visible, "__func__", None)
            is getattr(owner, "_native_sight_callback", None)
            and (range_check is None or native_range is not None)
        ):
            base, bonus = native_range if native_range is not None else (radius, 0)
            offsets = kernels.native.nearest_many(
                _front_points(
                    points,
                    ax,
                    ay,
                    getattr(range_check, "_native_sight_heading", None),
                    getattr(range_check, "_native_sight_cosine_squared", 0.0),
                ),
                ax,
                ay,
                animal_id,
                radius,
                k,
                *maps,
                owner.width,
                owner.height,
                base,
                bonus,
            )
            features = []
            for dx, dy in offsets:
                features.extend(
                    (dx / radius, dy / radius, math.hypot(dx, dy) / radius, 1.0)
                )
            features.extend([0.0] * (4 * k - len(features)))
            self.results[animal_id] = key, features
            return features
        radius_squared = radius * radius
        candidates = []
        for order, other_id, x, y in points:
            if other_id == animal_id:
                continue
            squared = (x - ax) ** 2 + (y - ay) ** 2
            if squared <= radius_squared:
                candidates.append((squared, order, x, y))
        # Sorting before ray tests avoids checking targets beyond the kth
        # visible one. Stable insertion-order ties match nearest(k=1).
        candidates.sort()
        features = []
        start = ax, ay
        for squared, order, x, y in candidates:
            end = x, y
            if range_check is not None and not range_check(start, end, squared):
                continue
            if not visible(start, end):
                continue
            dx, dy = x - ax, y - ay
            features.extend(
                (dx / radius, dy / radius, math.hypot(dx, dy) / radius, 1.0)
            )
            if len(features) == 4 * k:
                break
        features.extend([0.0] * (4 * k - len(features)))
        self.results[animal_id] = key, features
        return features
