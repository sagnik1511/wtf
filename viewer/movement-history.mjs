// Current combined/species grids share prior positions and one frame pass.
// Rewinds reuse the loaded chunk; distant seeks stream one prefix, never a grid archive.
const positions = (frame, species = null) => {
  const result = new Map();
  for (const animal of frame.animals)
    if (species === null || animal.species === species)
      result.set(animal.id, [animal.x, animal.y, animal.species]);
  return result;
};
export class MovementHistory {
  reset(width, height, species = null, speciesNames = []) {
    this.request = (this.request ?? 0) + 1;
    this.map = {
      width,
      height,
      species,
      counts: new Uint32Array(width * height),
      arrivals: 0,
      maximum: 0,
      frame_index: -1,
      tick: null,
      metric: 'changed_position_arrivals',
    };
    this.map.by_species = Object.fromEntries(
      speciesNames.map((name) => [
        name,
        {
          width,
          height,
          species: name,
          counts: new Uint32Array(width * height),
          arrivals: 0,
          maximum: 0,
          frame_index: -1,
          tick: null,
          metric: 'changed_position_arrivals',
        },
      ]),
    );
    this.maps = [this.map, ...Object.values(this.map.by_species)];
    this.previous = new Map();
    this.previousTick = null;
  }
  bump(x, y, sign, species) {
    this.bumpMap(this.map, x, y, sign);
    if (Object.hasOwn(this.map.by_species, species))
      this.bumpMap(this.map.by_species[species], x, y, sign);
  }
  bumpMap(m, x, y, sign) {
    if (
      !Number.isInteger(x) ||
      !Number.isInteger(y) ||
      x < 0 ||
      y < 0 ||
      x >= m.width ||
      y >= m.height
    )
      return;
    const i = y * m.width + x,
      next = m.counts[i] + sign;
    if (next < 0) throw new Error('Movement heatmap rewind underflow');
    m.counts[i] = next;
    m.arrivals += sign;
    if (sign > 0) m.maximum = Math.max(m.maximum, next);
  }
  transition(prior, frame, sign) {
    const current = positions(frame, this.map.species);
    const living =
      this.map.species === null
        ? current
        : new Set(frame.animals.map((a) => a.id));
    for (const [id, [x, y, species]] of current) {
      const old = prior.get(id);
      if (old && (old[0] !== x || old[1] !== y)) this.bump(x, y, sign, species);
    }
    for (const event of frame.events ?? []) {
      const old = prior.get(event.actor),
        p = event.position;
      if (
        event.kind === 'died' &&
        old &&
        !living.has(event.actor) &&
        Array.isArray(p) &&
        (old[0] !== p[0] || old[1] !== p[1])
      )
        this.bump(p[0], p[1], sign, old[2]);
    }
    return current;
  }
  async move(frames, start, index, loadPrefix) {
    const token = ++this.request,
      m = this.map,
      frame = frames[index - start];
    if (!frame) throw new Error('Movement replay frame unavailable');
    if (index === m.frame_index) return;
    if (index > m.frame_index && m.frame_index >= start - 1) {
      for (let i = m.frame_index + 1; i <= index; i++) {
        const next = frames[i - start];
        this.previous =
          this.previousTick !== null && next.tick === this.previousTick + 1
            ? this.transition(this.previous, next, 1)
            : positions(next, m.species);
        this.previousTick = next.tick;
      }
    } else if (
      index < m.frame_index &&
      m.frame_index < start + frames.length &&
      index >= start
    ) {
      for (let i = m.frame_index; i > index; i--) {
        const prior = frames[i - 1 - start],
          next = frames[i - start];
        if (next.tick === prior.tick + 1)
          this.transition(positions(prior, m.species), next, -1);
      }
      for (const map of this.maps)
        map.maximum = map.counts.reduce((max, n) => Math.max(max, n), 0);
      this.previous = positions(frame, m.species);
      this.previousTick = frame.tick;
    } else if (index === 0) {
      for (const map of this.maps) {
        map.counts.fill(0);
        map.arrivals = map.maximum = 0;
      }
      this.previous = positions(frame, m.species);
      this.previousTick = frame.tick;
    } else {
      const data = await loadPrefix(index);
      if (token !== this.request) return;
      const valid = (value, map) =>
        value &&
        (value.species ?? null) === map.species &&
        value.width === map.width &&
        value.height === map.height &&
        Array.isArray(value.counts) &&
        value.counts.length === map.counts.length &&
        value.counts.every(
          (n) => Number.isInteger(n) && n >= 0 && n <= 4294967295,
        );
      if (
        data.frame_index !== index ||
        data.tick !== frame.tick ||
        !valid(data, m) ||
        !Object.entries(m.by_species).every(
          ([name, map]) =>
            Object.hasOwn(data.by_species ?? {}, name) &&
            valid(data.by_species[name], map),
        )
      )
        throw new Error('Invalid movement heatmap prefix');
      for (const map of this.maps) {
        const source = map === m ? data : data.by_species[map.species];
        map.counts.set(source.counts);
        map.arrivals = map.counts.reduce((a, b) => a + b, 0);
        map.maximum = map.counts.reduce((a, b) => Math.max(a, b), 0);
      }
      this.previous = positions(frame, m.species);
      this.previousTick = frame.tick;
    }
    for (const map of this.maps) {
      map.frame_index = index;
      map.tick = frame.tick;
    }
  }
}
