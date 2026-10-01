import type { GraphNode, Paper } from "../types/paper";

export const MIN_EDGE_SIMILARITY = 0.35;
const MIN_EDGE_WIDTH = 1;
const EDGE_WIDTH_RANGE = 7;

export interface PaperDisplayRank {
  paper: Paper;
  queryRelevance: number;
  originalRank: number;
}

export function comparePaperDisplayRank(
  left: PaperDisplayRank,
  right: PaperDisplayRank,
): number {
  if (left.queryRelevance !== right.queryRelevance) {
    return left.queryRelevance > right.queryRelevance ? -1 : 1;
  }

  const originalRankOrder = left.originalRank - right.originalRank;
  if (originalRankOrder !== 0) return originalRankOrder;

  if (left.paper.id < right.paper.id) return -1;
  if (left.paper.id > right.paper.id) return 1;
  return 0;
}

export function orderPapersByQueryRelevance(
  papers: Paper[],
  nodes: GraphNode[],
): Paper[] {
  const relevanceByPaperId = new Map(
    nodes.map((node) => [node.paper_id, node.query_relevance]),
  );

  return papers
    .map((paper, originalRank): PaperDisplayRank => ({
      paper,
      originalRank,
      queryRelevance: relevanceByPaperId.get(paper.id) ?? Number.NEGATIVE_INFINITY,
    }))
    .sort(comparePaperDisplayRank)
    .map(({ paper }) => paper);
}

export function edgeWidth(weight: number): number {
  const normalized = Math.min(
    1,
    Math.max(0, (weight - MIN_EDGE_SIMILARITY) / (1 - MIN_EDGE_SIMILARITY)),
  );
  return MIN_EDGE_WIDTH + EDGE_WIDTH_RANGE * normalized ** 2;
}
