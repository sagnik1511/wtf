// Derive lightweight visual markers from recorded events. No simulation needed.
export const KILL_HOLD_FRAMES = 14;
const DIRECTIONS = [[0,-1],[1,0],[0,1],[-1,0]];

export function recentKills(frames, chunkStart, localIndex, carried = []) {
  const currentIndex = chunkStart + localIndex;
  const markers = [...carried];
  for (let i = Math.max(0, localIndex - KILL_HOLD_FRAMES + 1); i <= localIndex; i++) {
    const frame = frames[i];
    for (const event of frame.events ?? []) {
      if (event.kind !== 'kill') continue;
      const hunter = frame.animals.find(a => a.id === event.actor);
      const victim = frame.animals.find(a => a.id === event.target);
      // In existing recordings, killed prey have already been removed. An
      // attacker must face its adjacent victim and cannot move while attacking.
      const direction = hunter ? DIRECTIONS[hunter.heading] : null;
      const position = victim ? [victim.x, victim.y] : direction ? [hunter.x + direction[0], hunter.y + direction[1]] : null;
      markers.push({actor: event.actor, ...(hunter?.species?{attacker_species:hunter.species}:{}), target: event.target, tick: frame.tick, frame_index: chunkStart + i, position});
    }
  }
  const unique = new Map();
  for (const marker of markers) {
    const age = currentIndex - marker.frame_index;
    if (age >= 0 && age < KILL_HOLD_FRAMES) unique.set(`${marker.frame_index}:${marker.actor}:${marker.target}`, {...marker, age});
  }
  // Bound transient visual data even in very crowded recordings.
  return [...unique.values()].slice(-32);
}
