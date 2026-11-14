import { recentNaturalDeaths } from './death-highlights.mjs';
import { recentKills } from './kill-highlights.mjs';
// File-format reader only: no simulation, model weights, or learning code.
// One decoded chunk is retained, regardless of total recording duration.
export class Playback {
  constructor(loadJSON) {
    this.loadJSON = loadJSON;
    this.frames = [];
    this.metrics = [];
    this.killCarry = [];
    this.deathCarry = [];
    this.previousAgents = [];
    this.previousTick = null;
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
  async changeRun(run, selected = null) {
    const entry = this.runs.find(entry => entry.run === Number(run));
    if (!entry) throw new Error('Recorded run not found');
    this.run = entry;
    this.metrics = [];
    const metrics = await this.loadJSON(`/api/recordings/${encodeURIComponent(this.id)}/runs/${entry.run}/metrics`);
    if (!Array.isArray(metrics) || metrics.length > 2048) throw new Error('Invalid run metrics');
    this.metrics = metrics;
    this.frames = [];
    this.killCarry = [];
    this.deathCarry = [];
    this.previousAgents = [];
    this.previousTick = null;
    this.chunkIndex = -1;
    return this.seek(0, selected);
  }
  async seek(index, selected = null) {
    if (!Number.isInteger(index) || index < 0 || index >= this.run.frame_count) throw new Error('Frame outside recording');
    const size = this.metadata.recording.frames_per_chunk;
    const chunkIndex = Math.floor(index / size);
    if (chunkIndex !== this.chunkIndex) {
      const sequential = index === this.frameIndex + 1 && chunkIndex === this.chunkIndex + 1 && this.frames.length;
      const previousSnapshot = sequential ? this.snapshot() : null;
      this.killCarry = previousSnapshot?.kill_highlights ?? [];
      this.deathCarry = previousSnapshot?.natural_death_highlights ?? [];
      this.previousTick = sequential ? this.frames.at(-1).tick : null;
      this.previousAgents = sequential ? this.frames.at(-1).animals.map(({id,x,y,heading,species})=>({id,x,y,heading,species})) : [];
      this.frames = []; // Release the previous chunk before decoding another.
      this.chunkIndex = -1;
      if (!sequential && chunkIndex > 0) {
        // Retain only small position records, not another world chunk. This
        // also locates first-frame deaths after jumping directly to a chunk.
        const previous = await this.loadJSON(`/api/recordings/${encodeURIComponent(this.id)}/runs/${this.run.run}/chunks/${chunkIndex-1}`);
        if (!Array.isArray(previous) || !previous.length || previous.length > size) throw new Error('Invalid recording chunk');
        this.previousTick = previous.at(-1).tick;
        this.previousAgents = previous.at(-1).animals.map(({id,x,y,heading,species})=>({id,x,y,heading,species}));
      }
      const frames = await this.loadJSON(`/api/recordings/${encodeURIComponent(this.id)}/runs/${this.run.run}/chunks/${chunkIndex}`);
      if (!Array.isArray(frames) || frames.length > size || !frames.length) throw new Error('Invalid recording chunk');
      this.frames = frames;
      this.chunkIndex = chunkIndex;
    }
    this.frameIndex = index;
    return this.snapshot(selected);
  }
  get atEnd() {
    return this.frameIndex === this.run.frame_count - 1 && this.run === this.runs[this.runs.length - 1];
  }
  async advance(selected = null) {
    if (this.frameIndex < this.run.frame_count - 1) return this.seek(this.frameIndex + 1, selected);
    const next = this.runs[this.runs.indexOf(this.run) + 1];
    return next ? this.changeRun(next.run, selected) : this.snapshot(selected);
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
    const history = this.metrics.filter(metric => metric.tick <= frame.tick);
    // Keep the exact current point even between sampled metrics or after seek.
    if (!history.length || history.at(-1).tick !== frame.tick) {
      history.push({tick: frame.tick, food: frame.metrics.food, populations: frame.metrics.populations});
    }
    return {
      ...frame, selected: diagnostic, width: c.world.width, height: c.world.height,
      resource_capacity: c.resources.capacity_per_patch, obstacles: this.run.obstacles,
      species: c.species, name: this.metadata.name, mode: this.metadata.mode,
      run: this.run.run, total_runs: c.experiment?.runs ?? this.runs.length, done: this.atEnd,
      // Full-run lightweight metrics survive chunk changes; no future points.
      history,
      natural_death_highlights: recentNaturalDeaths(this.frames, this.chunkIndex * this.metadata.recording.frames_per_chunk, local, this.deathCarry, this.previousAgents, this.previousTick),
      kill_highlights: recentKills(this.frames, this.chunkIndex * this.metadata.recording.frames_per_chunk, local, this.killCarry),
      frame_index: this.frameIndex, frame_count: this.run.frame_count
    };
  }
}
