import { recentKills } from './kill-highlights.mjs';
// File-format reader only: no simulation, model weights, or learning code.
// One decoded chunk is retained, regardless of total recording duration.
export class Playback {
  constructor(loadJSON) {
    this.loadJSON = loadJSON;
    this.frames = [];
    this.killCarry = [];
    this.chunkIndex = -1;
    this.frameIndex = 0;
  }
  async open(id) {
    const metadata = await this.loadJSON(`/api/recordings/${encodeURIComponent(id)}`);
    if (metadata.recording_version !== 1) throw new Error('Unsupported recording version');
    const runs = metadata.runs.filter(run => run.frame_count > 0);
    if (!runs.length) throw new Error('This recording has no committed frames yet');
    this.id = id;
    this.metadata = metadata;
    this.runs = runs;
    return this.changeRun(runs[0].run);
  }
  async changeRun(run) {
    const entry = this.runs.find(entry => entry.run === Number(run));
    if (!entry) throw new Error('Recorded run not found');
    this.run = entry;
    this.frames = [];
    this.killCarry = [];
    this.chunkIndex = -1;
    return this.seek(0);
  }
  async seek(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.run.frame_count) throw new Error('Frame outside recording');
    const size = this.metadata.recording.frames_per_chunk;
    const chunkIndex = Math.floor(index / size);
    if (chunkIndex !== this.chunkIndex) {
      const sequential = index === this.frameIndex + 1 && chunkIndex === this.chunkIndex + 1 && this.frames.length;
      this.killCarry = sequential ? this.snapshot().kill_highlights : [];
      this.frames = []; // Release the previous chunk before decoding another.
      this.chunkIndex = -1;
      const frames = await this.loadJSON(`/api/recordings/${encodeURIComponent(this.id)}/runs/${this.run.run}/chunks/${chunkIndex}`);
      if (!Array.isArray(frames) || frames.length > size || !frames.length) throw new Error('Invalid recording chunk');
      this.frames = frames;
      this.chunkIndex = chunkIndex;
    }
    this.frameIndex = index;
    return this.snapshot();
  }
  get atEnd() {
    return this.frameIndex === this.run.frame_count - 1 && this.run === this.runs[this.runs.length - 1];
  }
  async advance() {
    if (this.frameIndex < this.run.frame_count - 1) return this.seek(this.frameIndex + 1);
    const next = this.runs[this.runs.indexOf(this.run) + 1];
    return next ? this.changeRun(next.run) : this.snapshot();
  }
  snapshot(selected = null) {
    const local = this.frameIndex % this.metadata.recording.frames_per_chunk;
    const frame = this.frames[local];
    if (!frame) throw new Error('Recorded frame missing');
    let diagnostic;
    for (let i = local; i >= 0; i--) {
      if (this.frames[i].diagnostics) {
        diagnostic = this.frames[i].diagnostics[String(selected)];
        break;
      }
    }
    const c = this.metadata.config;
    return {
      ...frame, selected: diagnostic, width: c.world.width, height: c.world.height,
      resource_capacity: c.resources.capacity_per_patch, obstacles: this.run.obstacles,
      species: c.species, name: this.metadata.name, mode: this.metadata.mode,
      run: this.run.run, total_runs: c.experiment?.runs ?? this.runs.length, done: this.atEnd,
      // Chart data is also bounded by the current chunk; seeking is independent.
      history: this.frames.slice(0, local + 1).map(frame => frame.metrics),
      kill_highlights: recentKills(this.frames, this.chunkIndex * this.metadata.recording.frames_per_chunk, local, this.killCarry),
      frame_index: this.frameIndex, frame_count: this.run.frame_count
    };
  }
}
