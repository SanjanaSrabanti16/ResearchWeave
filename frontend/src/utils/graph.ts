export const MIN_EDGE_SIMILARITY = 0.35;
const MIN_EDGE_WIDTH = 1;
const EDGE_WIDTH_RANGE = 7;

export function edgeWidth(weight: number): number {
  const normalized = Math.min(
    1,
    Math.max(0, (weight - MIN_EDGE_SIMILARITY) / (1 - MIN_EDGE_SIMILARITY)),
  );
  return MIN_EDGE_WIDTH + EDGE_WIDTH_RANGE * normalized ** 2;
}
