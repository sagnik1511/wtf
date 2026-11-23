import { recentFeedings, FEED_HOLD_FRAMES } from './feeding-highlights.mjs';
import { MovementHistory } from './movement-history.mjs';
import { CollectiveHistory } from './collective-history.mjs';
import { recentContests } from './competition.mjs';
import { decodeChunk } from './replay-codec.mjs';
import { recentNaturalDeaths } from './death-highlights.mjs';
import { recentBirths } from './birth-highlights.mjs';
import { recentKills } from './kill-highlights.mjs';
// File-format reader only: no simulation, model weights, or learning code.
// One decoded chunk is retained, regardless of total recording duration.
export class Playback {
  constructor(
    loadJSON,
    now = () => Date.now(),
    { movement = false, collective = false } = {},
  ) {
    this.loadJSON = loadJSON;
    this.movement = movement ? new MovementHistory() : null;
    this.collectiveEnabled = collective;
    this.collective = null;
    this.now = now;
    this.lastMetadataRefresh = -Infinity;
    this.frames = [];
    this.metrics = [];
    this.killCarry = [];
    this.deathCarry = [];
    this.birthCarry = [];
    this.contestCarry = [];
    this.feedingCarry = [];
    this.previousAgents = [];
    this.previousTick = null;
    this.chunkIndex = -1;
    this.frameIndex = 0;
  }
  async open(id) {
    const metadata = await this.loadJSON(
      `/api/recordings/${encodeURIComponent(id)}`,
    );
    if (metadata.recording_version !== 1)
      throw new Error('Unsupported recording version');
    const runs = metadata.runs.filter((run) => run.frame_count > 0);
    if (!runs.length)
      throw new Error('This recording has no committed frames yet');
    this.id = id;
    this.metadata = metadata;
    this.runs = runs;
    this.lastMetadataRefresh = this.now();
    return this.changeRun(runs[0].run);
  }
  async changeRun(run, selected = null) {
    const entry = this.runs.find((entry) => entry.run === Number(run));
    if (!entry) throw new Error('Recorded run not found');
    this.run = entry;
    this.collective =
      this.collectiveEnabled &&
      Object.values(this.metadata.config.species ?? {}).some((spec) => {
        const b = this.metadata.config.brains?.[spec.brain];
        return (
          b && (b.social_observation !== undefined || b.consequence_observation)
        );
      })
        ? new CollectiveHistory()
        : null;
    if (this.collective) {
      this.collective.reset(
        this.metadata.config,
        this.metadata.recording.frames_per_chunk,
      );
    }
    this.movement?.reset(
      this.metadata.config.world.width,
      this.metadata.config.world.height,
    );
    this.metrics = [];
    const metrics = await this.loadJSON(
      `/api/recordings/${encodeURIComponent(this.id)}/runs/${entry.run}/metrics`,
    );
    if (!Array.isArray(metrics) || metrics.length > 2048)
      throw new Error('Invalid run metrics');
    this.metrics = metrics;
    this.frames = [];
    this.killCarry = [];
    this.deathCarry = [];
    this.birthCarry = [];
    this.contestCarry = [];
    this.feedingCarry = [];
    this.previousAgents = [];
    this.previousTick = null;
    this.chunkIndex = -1;
    return this.seek(0, selected);
  }
  async seek(index, selected = null) {
    if (!Number.isInteger(index) || index < 0 || index >= this.run.frame_count)
      throw new Error('Frame outside recording');
    const size = this.metadata.recording.frames_per_chunk;
    const chunkIndex = Math.floor(index / size);
    if (chunkIndex !== this.chunkIndex) {
      const sequential =
        index === this.frameIndex + 1 &&
        chunkIndex === this.chunkIndex + 1 &&
        this.frames.length;
      const previousSnapshot = sequential ? this.snapshot() : null;
      this.killCarry = previousSnapshot?.kill_highlights ?? [];
      this.deathCarry = previousSnapshot?.natural_death_highlights ?? [];
      this.birthCarry = previousSnapshot?.birth_highlights ?? [];
      this.contestCarry = previousSnapshot?.contest_highlights ?? [];
      this.feedingCarry = previousSnapshot?.feeding_highlights ?? [];
      this.previousTick = sequential ? this.frames.at(-1).tick : null;
      this.previousAgents = sequential
        ? this.frames.at(-1).animals.map(({ id, x, y, heading, species }) => ({
            id,
            x,
            y,
            heading,
            species,
          }))
        : [];
      this.frames = []; // Release the previous chunk before decoding another.
      this.chunkIndex = -1;
      if (!sequential && chunkIndex > 0) {
        // Retain only small position records, not another world chunk. This
        // also locates first-frame deaths after jumping directly to a chunk.
        const previous = decodeChunk(
          await this.loadJSON(
            `/api/recordings/${encodeURIComponent(this.id)}/runs/${this.run.run}/chunks/${chunkIndex - 1}`,
          ),
        );
        if (
          !Array.isArray(previous) ||
          !previous.length ||
          previous.length > size
        )
          throw new Error('Invalid recording chunk');
        // Very small legacy chunks can span several chunks during a short cue.
        for (
          let c = Math.max(
            0,
            chunkIndex - Math.ceil((FEED_HOLD_FRAMES - 1) / size),
          );
          c < chunkIndex - 1;
          c++
        ) {
          const older = decodeChunk(
            await this.loadJSON(
              `/api/recordings/${encodeURIComponent(this.id)}/runs/${this.run.run}/chunks/${c}`,
            ),
          );
          if (!Array.isArray(older) || !older.length || older.length > size)
            throw new Error('Invalid recording chunk');
          this.feedingCarry = recentFeedings(
            older,
            c * size,
            older.length - 1,
            this.feedingCarry,
          );
        }
        this.feedingCarry = recentFeedings(
          previous,
          (chunkIndex - 1) * size,
          previous.length - 1,
          this.feedingCarry,
        );
        this.previousTick = previous.at(-1).tick;
        this.previousAgents = previous
          .at(-1)
          .animals.map(({ id, x, y, heading, species }) => ({
            id,
            x,
            y,
            heading,
            species,
          }));
      }
      const frames = decodeChunk(
        await this.loadJSON(
          `/api/recordings/${encodeURIComponent(this.id)}/runs/${this.run.run}/chunks/${chunkIndex}`,
        ),
      );
      if (!Array.isArray(frames) || frames.length > size || !frames.length)
        throw new Error('Invalid recording chunk');
      this.frames = frames;
      this.chunkIndex = chunkIndex;
    }
    if (this.movement)
      await this.movement.move(this.frames, chunkIndex * size, index, (frame) =>
        this.loadJSON(
          `/api/recordings/${encodeURIComponent(this.id)}/runs/${this.run.run}/movement/${frame}`,
        ),
      );
    if (this.collective)
      await this.collective.move(
        this.frames,
        chunkIndex * size,
        index,
        async (chunk) =>
          decodeChunk(
            await this.loadJSON(
              `/api/recordings/${encodeURIComponent(this.id)}/runs/${this.run.run}/chunks/${chunk}`,
            ),
          ),
      );
    this.frameIndex = index;
    return this.snapshot(selected);
  }
  async refreshMetadata(selected = null, force = false) {
    if (!force && this.now() - this.lastMetadataRefresh < 2000) return;
    const metadata = await this.loadJSON(
      `/api/recordings/${encodeURIComponent(this.id)}`,
    );
    if (metadata.recording_version !== 1)
      throw new Error('Unsupported recording version');
    const runs = metadata.runs.filter((run) => run.frame_count > 0);
    const entry = runs.find((run) => run.run === this.run.run);
    if (!entry) throw new Error('Selected run is no longer available');
    const oldCount = this.run.frame_count;
    this.metadata = metadata;
    this.runs = runs;
    this.run = entry;
    this.lastMetadataRefresh = this.now();
    if (entry.frame_count !== oldCount) {
      const metrics = await this.loadJSON(
        `/api/recordings/${encodeURIComponent(this.id)}/runs/${entry.run}/metrics`,
      );
      if (!Array.isArray(metrics) || metrics.length > 2048)
        throw new Error('Invalid run metrics');
      this.metrics = metrics;
      const size = metadata.recording.frames_per_chunk;
      // Normal committed chunks are immutable. Reload a previously partial
      // last chunk if an exporter has extended it in place.
      if (
        oldCount % size &&
        this.chunkIndex === Math.floor((oldCount - 1) / size)
      ) {
        this.chunkIndex = -1;
        await this.seek(this.frameIndex, selected);
      }
    }
  }
  get atEnd() {
    return (
      this.frameIndex === this.run.frame_count - 1 &&
      this.run.run === this.runs.at(-1).run &&
      this.metadata.status !== 'writing'
    );
  }
  async advance(selected = null) {
    if (this.frameIndex < this.run.frame_count - 1)
      return this.seek(this.frameIndex + 1, selected);
    let next =
      this.runs[this.runs.findIndex((run) => run.run === this.run.run) + 1];
    if (!next && this.metadata.status === 'writing') {
      await this.refreshMetadata(selected);
      if (this.frameIndex < this.run.frame_count - 1)
        return this.seek(this.frameIndex + 1, selected);
      next =
        this.runs[this.runs.findIndex((run) => run.run === this.run.run) + 1];
    }
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
    const history = this.metrics.filter((metric) => metric.tick <= frame.tick);
    // Keep the exact current point even between sampled metrics or after seek.
    if (!history.length || history.at(-1).tick !== frame.tick) {
      history.push({
        tick: frame.tick,
        food: frame.metrics.food,
        populations: frame.metrics.populations,
      });
    }
    return {
      ...frame,
      selected: diagnostic,
      collective: this.collective?.totals.view(selected) ?? null,
      width: c.world.width,
      height: c.world.height,
      resource_capacity: c.resources.capacity_per_patch,
      obstacles: this.run.obstacles,
      ...(c.terrain ? { terrain: c.terrain } : {}),
      species: c.species,
      ...(c.day_night ? { day_night: c.day_night } : {}),
      name: this.metadata.name,
      mode: this.metadata.mode,
      run: this.run.run,
      total_runs: c.experiment?.runs ?? this.runs.length,
      done: this.atEnd,
      // Full-run lightweight metrics survive chunk changes; no future points.
      history,
      feeding_highlights: recentFeedings(
        this.frames,
        this.chunkIndex * this.metadata.recording.frames_per_chunk,
        local,
        this.feedingCarry,
      ),
      contest_highlights: recentContests(
        this.frames,
        this.chunkIndex * this.metadata.recording.frames_per_chunk,
        local,
        this.contestCarry,
      ),
      birth_highlights: recentBirths(
        this.frames,
        this.chunkIndex * this.metadata.recording.frames_per_chunk,
        local,
        this.birthCarry,
      ),
      natural_death_highlights: recentNaturalDeaths(
        this.frames,
        this.chunkIndex * this.metadata.recording.frames_per_chunk,
        local,
        this.deathCarry,
        this.previousAgents,
        this.previousTick,
      ),
      kill_highlights: recentKills(
        this.frames,
        this.chunkIndex * this.metadata.recording.frames_per_chunk,
        local,
        this.killCarry,
      ),
      waiting:
        this.metadata.status === 'writing' &&
        this.frameIndex === this.run.frame_count - 1 &&
        this.run.run === this.runs.at(-1).run,
      frame_index: this.frameIndex,
      frame_count: this.run.frame_count,
    };
  }
}
