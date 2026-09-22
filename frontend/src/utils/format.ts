export function formatScore(score: number | null): string {
  return score === null ? "—" : score.toFixed(4);
}
