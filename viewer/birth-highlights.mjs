// Birth positions come directly from saved newborns; no simulation needed.
export const BIRTH_HOLD_FRAMES = 14;

export function recentBirths(frames, chunkStart, localIndex, carried = []) {
  const currentIndex = chunkStart + localIndex, markers = [...carried];
  for (let i = Math.max(0, localIndex - BIRTH_HOLD_FRAMES + 1); i <= localIndex; i++) {
    const frame = frames[i];
    for (const event of frame.events ?? []) {
      if (event.kind !== 'born') continue;
      const child = frame.animals.find(a => a.id === event.target);
      markers.push({actor: event.actor, target: event.target, species: child?.species,
        tick: frame.tick, frame_index: chunkStart + i,
        position: child ? [child.x, child.y] : null});
    }
  }
  const unique = new Map();
  for (const marker of markers) {
    const age = currentIndex - marker.frame_index;
    if (age >= 0 && age < BIRTH_HOLD_FRAMES) {
      unique.set(`${marker.frame_index}:${marker.target}`, {...marker, age});
    }
  }
  return [...unique.values()].slice(-32);
}
