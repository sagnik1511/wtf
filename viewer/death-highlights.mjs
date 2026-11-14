import { KILL_HOLD_FRAMES } from './kill-highlights.mjs';
const DIRECTIONS = [[0,-1],[1,0],[0,1],[-1,0]];

// Deaths are logged after bodies are removed. Previous positions plus recorded
// successful movement recover their final locations without simulating a step.
export function recentNaturalDeaths(frames, chunkStart, localIndex, carried = [], previousAgents = [], previousTick = null) {
  const currentIndex = chunkStart + localIndex, markers = [...carried];
  for (let i = Math.max(0, localIndex - KILL_HOLD_FRAMES + 1); i <= localIndex; i++) {
    const frame = frames[i], events = frame.events ?? [];
    const killed = new Set(events.filter(e => e.kind === 'kill').map(e => e.target));
    for (const event of events) {
      if (event.kind !== 'died' || event.cause === 'predation' || killed.has(event.actor)) continue;
      const current = frame.animals.find(a => a.id === event.actor);
      const previous = (i > 0 ? frames[i - 1].animals : previousAgents).find(a => a.id === event.actor);
      const animal = current ?? previous;
      const precedingTick = i > 0 ? frames[i - 1].tick : previousTick;
      const adjacent = precedingTick === frame.tick - 1;
      let position = current ? [current.x, current.y] : previous && adjacent ? [previous.x, previous.y] : null;
      if (!current && previous && adjacent && events.some(e => e.kind === 'moved' && e.actor === event.actor)) {
        const [dx, dy] = DIRECTIONS[previous.heading];
        position = [previous.x + dx, previous.y + dy];
      }
      if (Array.isArray(event.position) && event.position.length === 2 && event.position.every(Number.isFinite)) position = event.position;
      markers.push({actor: event.actor, species: animal?.species, cause: event.cause || 'natural', tick: frame.tick, frame_index: chunkStart + i, position});
    }
  }
  const unique = new Map();
  for (const marker of markers) {
    const age = currentIndex - marker.frame_index;
    if (age >= 0 && age < KILL_HOLD_FRAMES) unique.set(`${marker.frame_index}:${marker.actor}`, {...marker, age});
  }
  return [...unique.values()].slice(-32);
}
