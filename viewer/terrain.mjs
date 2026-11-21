// Static map metadata is decoded once per recording, never once per frame/chunk.
export function terrainView(definition) {
  if (!definition?.rows) return null;
  const tiles = [],
    cells = new Map();
  let maxElevation = 1;
  for (let y = 0; y < definition.rows.length; y++)
    for (let x = 0; x < definition.rows[y].length; x++) {
      const cell = definition.legend[definition.rows[y][x]];
      const tile = {
        x,
        y,
        elevation: cell.elevation ?? 0,
        depth: cell.water_depth ?? 0,
        flow: cell.flow ?? [0, 0],
      };
      if (tile.elevation || tile.depth) {
        tiles.push(tile);
        cells.set(y * definition.width + x, tile);
      }
      maxElevation = Math.max(maxElevation, tile.elevation);
    }
  return {
    name: definition.name,
    width: definition.width,
    tiles,
    cells,
    maxElevation,
    hasWater: tiles.some((t) => t.depth > 0),
    hasElevation: tiles.some((t) => t.elevation > 0),
  };
}

export function drawTerrain(ctx, terrain, cw, ch, light = false) {
  if (!terrain) return;
  for (const t of terrain.tiles) {
    const x = t.x * cw,
      y = t.y * ch;
    ctx.fillStyle = t.depth
      ? `hsl(207, 65%, ${(light ? 65 : 22) + Math.min(t.depth, 3) * (light ? -5 : 5)}%)`
      : `hsl(${85 - (50 * t.elevation) / terrain.maxElevation}, 25%, ${(light ? 84 : 19) + ((light ? -20 : 14) * t.elevation) / terrain.maxElevation}%)`;
    ctx.fillRect(x, y, cw, ch);
    if (t.depth && (t.x + t.y) % 3 === 0) {
      const [dx, dy] = t.flow,
        n = Math.hypot(dx, dy);
      if (n) {
        const ux = dx / n,
          uy = dy / n,
          r = Math.min(cw, ch) * 0.32,
          cx = x + cw / 2,
          cy = y + ch / 2;
        ctx.strokeStyle = light ? '#145b8fb0' : '#89d4ff90';
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.moveTo(cx - ux * r, cy - uy * r);
        ctx.lineTo(cx + ux * r, cy + uy * r);
        ctx.moveTo(cx + ux * r, cy + uy * r);
        ctx.lineTo(
          cx + ux * r * 0.35 - uy * r * 0.45,
          cy + uy * r * 0.35 + ux * r * 0.45,
        );
        ctx.moveTo(cx + ux * r, cy + uy * r);
        ctx.lineTo(
          cx + ux * r * 0.35 + uy * r * 0.45,
          cy + uy * r * 0.35 - ux * r * 0.45,
        );
        ctx.stroke();
      }
    } else if (!t.depth && t.x % 4 === 0 && t.y % 4 === 0) {
      ctx.fillStyle = light ? '#4d492dce' : '#f0dfba75';
      ctx.font = `${Math.max(8, Math.min(cw, ch) * 0.4)}px system-ui`;
      ctx.textAlign = 'center';
      ctx.fillText(String(t.elevation), x + cw / 2, y + ch * 0.7);
    }
  }
}
