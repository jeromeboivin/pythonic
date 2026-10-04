// pure logic: no DOM, no Qt
export function dragToValue(start, dyPx, min, max, fine = false) {
  const range = max - min, px = fine ? 2000 : 200;
  return Math.min(max, Math.max(min, start - (dyPx / px) * range));
}
