// Bounded, replay-only highlights; no additional recording data or playback pause.
export const CONTEST_HOLD_FRAMES = 12;
export function contestLabel(cause) {
  return (
    {
      food: 'Food priority',
      blocked_path: 'Path contest',
      aggression: 'Aggression',
    }[cause] ?? 'Contest'
  );
}
export function recentContests(frames, chunkStart, localIndex, carried = []) {
  const markers = [...carried];
  for (
    let i = Math.max(0, localIndex - CONTEST_HOLD_FRAMES + 1);
    i <= localIndex;
    i++
  ) {
    const frame = frames[i];
    for (const event of frame.events ?? []) {
      if (event.kind !== 'contest_won') continue;
      const lost = frame.events.find(
        (e) =>
          e.kind === 'contest_lost' &&
          e.actor === event.target &&
          e.target === event.actor,
      );
      markers.push({
        ...event,
        loser_position: lost?.position,
        tick: frame.tick,
        frame_index: chunkStart + i,
      });
    }
  }
  const unique = new Map();
  for (const marker of markers) {
    const age = chunkStart + localIndex - marker.frame_index;
    if (age >= 0 && age < CONTEST_HOLD_FRAMES)
      unique.set(`${marker.frame_index}:${marker.actor}:${marker.target}`, {
        ...marker,
        age,
      });
  }
  return [...unique.values()].slice(-32);
}
export function drawContests(ctx, markers, cw, ch, dark) {
  for (const marker of markers ?? []) {
    if (
      !Array.isArray(marker.position) ||
      !Array.isArray(marker.loser_position)
    )
      continue;
    const [x, y] = marker.position.map((v, i) => (v + 0.5) * (i ? ch : cw));
    const [tx, ty] = marker.loser_position.map(
      (v, i) => (v + 0.5) * (i ? ch : cw),
    );
    ctx.save();
    ctx.globalAlpha = Math.max(0.35, 1 - marker.age / CONTEST_HOLD_FRAMES);
    ctx.strokeStyle = dark ? '#ffd08a' : '#9a4100';
    ctx.lineWidth = Math.max(2, cw * 0.09);
    ctx.setLineDash([Math.max(2, cw * 0.15), Math.max(2, cw * 0.1)]);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const [px, py] of [
      [x, y],
      [tx, ty],
    ]) {
      ctx.beginPath();
      ctx.arc(px, py, Math.min(cw, ch) * 0.65, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.font = `bold ${Math.max(12, cw * 0.2)}px system-ui`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const label = `#${marker.actor} wins · −${marker.amount} HP`;
    ctx.lineWidth = 3;
    ctx.strokeStyle = dark ? '#111827' : '#ffffff';
    ctx.strokeText(label, (x + tx) / 2, Math.min(y, ty) - ch * 0.72);
    ctx.fillStyle = dark ? '#ffd08a' : '#9a4100';
    ctx.fillText(label, (x + tx) / 2, Math.min(y, ty) - ch * 0.72);
    ctx.restore();
  }
}
