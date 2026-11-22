// One reusable grid and prior positions. Rewinds reuse the loaded chunk; large
// jumps request a streamed prefix, never a per-frame grid archive.
const positions = (frame) =>
  new Map(frame.animals.map((a) => [a.id, [a.x, a.y]]));
export class MovementHistory {
  reset(width, height) {
    this.request = (this.request ?? 0) + 1;
    this.map = {
      width,
      height,
      counts: new Uint32Array(width * height),
      arrivals: 0,
      maximum: 0,
      frame_index: -1,
      tick: null,
      metric: 'changed_position_arrivals',
    };
    this.previous = new Map();
    this.previousTick = null;
  }
  bump(x, y, sign) {
    const m = this.map;
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
    const current = positions(frame);
    for (const [id, [x, y]] of current) {
      const old = prior.get(id);
      if (old && (old[0] !== x || old[1] !== y)) this.bump(x, y, sign);
    }
    for (const event of frame.events ?? []) {
      const old = prior.get(event.actor),
        p = event.position;
      if (
        event.kind === 'died' &&
        old &&
        !current.has(event.actor) &&
        Array.isArray(p) &&
        (old[0] !== p[0] || old[1] !== p[1])
      )
        this.bump(p[0], p[1], sign);
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
            : positions(next);
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
          this.transition(positions(prior), next, -1);
      }
      m.maximum = m.counts.reduce((max, n) => Math.max(max, n), 0);
      this.previous = positions(frame);
      this.previousTick = frame.tick;
    } else if (index === 0) {
      m.counts.fill(0);
      m.arrivals = m.maximum = 0;
      this.previous = positions(frame);
      this.previousTick = frame.tick;
    } else {
      const data = await loadPrefix(index);
      if (token !== this.request) return;
      if (
        data.frame_index !== index ||
        data.tick !== frame.tick ||
        data.width !== m.width ||
        data.height !== m.height ||
        !Array.isArray(data.counts) ||
        data.counts.length !== m.counts.length ||
        !data.counts.every(
          (n) => Number.isInteger(n) && n >= 0 && n <= 4294967295,
        )
      )
        throw new Error('Invalid movement heatmap prefix');
      m.counts.set(data.counts);
      m.arrivals = m.counts.reduce((a, b) => a + b, 0);
      m.maximum = m.counts.reduce((a, b) => Math.max(a, b), 0);
      this.previous = positions(frame);
      this.previousTick = frame.tick;
    }
    m.frame_index = index;
    m.tick = frame.tick;
  }
}
