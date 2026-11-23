// Cosmetic markers derived only from confirmed saved food gains.
export const FEED_HOLD_FRAMES = 4;
export function recentFeedings(frames, chunkStart, localIndex, carried = []) {
  const current = chunkStart + localIndex;
  const markers = new Map();
  function add(marker) {
    const age = current - marker.frame_index;
    if (age >= 0 && age < FEED_HOLD_FRAMES)
      markers.set(`${marker.frame_index}:${marker.actor}`, { ...marker, age });
  }
  for (const marker of carried) add(marker);
  for (
    let i = Math.max(0, localIndex - FEED_HOLD_FRAMES + 1);
    i <= localIndex;
    i++
  ) {
    const frame = frames[i];
    // Build a position lookup only when an event actually qualifies.
    let positions;
    for (const event of frame.events ?? []) {
      if (
        event.kind !== 'energy_gained' ||
        !(event.amount > 0) ||
        !['resource', 'competition_food'].includes(event.cause)
      )
        continue;
      positions ??= new Map(frame.animals.map((a) => [a.id, a]));
      const animal = positions.get(event.actor);
      if (animal)
        add({
          actor: event.actor,
          position: [animal.x, animal.y],
          frame_index: chunkStart + i,
          tick: frame.tick,
        });
    }
  }
  return [...markers.values()].slice(-64);
}
export function drawFeedings(
  ctx,
  markers,
  cw,
  ch,
  { light = false, reducedMotion = false } = {},
) {
  ctx.save();
  for (const marker of markers ?? []) {
    const age = reducedMotion ? 0 : marker.age;
    const x = (marker.position[0] + 0.78) * cw,
      y = (marker.position[1] + 0.22) * ch;
    const r = Math.min(cw, ch) * (0.09 + age * 0.016);
    ctx.globalAlpha = reducedMotion ? 1 : 1 - age / FEED_HOLD_FRAMES;
    ctx.fillStyle = light ? '#34784a' : '#a1c8b0';
    ctx.strokeStyle = light ? '#e9f3f8' : '#102632';
    ctx.lineWidth = Math.max(1, r * 0.3);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}
