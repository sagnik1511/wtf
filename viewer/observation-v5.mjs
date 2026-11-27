// Coordinate conversion for display only. Raw V5 vectors remain body-relative.
export const BASE_WIDTHS = { 1: 20, 2: 32, 3: 42, 4: 45, 5: 48 };
export const OPPORTUNITY_NAMES = [
  'Attack possible now',
  'Eat possible now',
  'Swim possible now',
];
export const BODY_SECTORS = [
  'here',
  'front',
  'front-right',
  'right',
  'back-right',
  'back',
  'back-left',
  'left',
  'front-left',
];
export function worldVector(forward, right, heading) {
  const [hx, hy] = heading;
  return [forward * hx - right * hy, forward * hy + right * hx];
}
export function worldScent(values, heading) {
  const offsets = [
    [0, 0],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
  ];
  const result = Array(9).fill(0);
  offsets.forEach(([x, y], i) => {
    const [wx, wy] = worldVector(-y, x, heading);
    result[offsets.findIndex(([a, b]) => a === wx && b === wy)] = values[i];
  });
  return result;
}
