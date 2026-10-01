import { describe, expect, it } from "vitest";

import type { GraphNode, Paper } from "../types/paper";
import {
  comparePaperDisplayRank,
  orderPapersByQueryRelevance,
} from "./graph";

function paper(id: string, semanticScore: number): Paper {
  return {
    id,
    title: `Paper ${id}`,
    abstract: null,
    authors: [],
    publication_year: null,
    publication_date: null,
    venue: null,
    doi: null,
    arxiv_id: null,
    openalex_id: null,
    semantic_scholar_id: null,
    url: null,
    pdf_url: null,
    citation_count: null,
    source_names: ["openalex"],
    semantic_score: semanticScore,
    reranker_score: semanticScore,
  };
}

function node(paperId: string, queryRelevance: number): GraphNode {
  return {
    paper_id: paperId,
    title: `Paper ${paperId}`,
    query_relevance: queryRelevance,
    node_weight: queryRelevance,
    node_radius: 20,
    information_completeness: 1,
    node_opacity: 1,
  };
}

describe("query-relevance paper display order", () => {
  it("sorts descending by canonical graph query relevance without mutating M1 results", () => {
    const searchOrder = [paper("m1-first", 0.91), paper("m1-second", 0.72), paper("m1-third", 0.54)];
    const originalSnapshot = structuredClone(searchOrder);

    const displayed = orderPapersByQueryRelevance(searchOrder, [
      node("m1-first", 0.61),
      node("m1-second", 0.72),
      node("m1-third", 0.96),
    ]);

    expect(displayed.map(({ id }) => id)).toEqual(["m1-third", "m1-second", "m1-first"]);
    expect(displayed[0]).toBe(searchOrder[2]);
    expect(searchOrder).toEqual(originalSnapshot);
    expect(searchOrder.map(({ semantic_score }) => semantic_score)).toEqual([0.91, 0.72, 0.54]);
  });

  it("uses original M1 order for equal query relevance", () => {
    const searchOrder = [paper("paper-z", 0.8), paper("paper-a", 0.7)];

    const displayed = orderPapersByQueryRelevance(searchOrder, [
      node("paper-z", 0.9),
      node("paper-a", 0.9),
    ]);

    expect(displayed.map(({ id }) => id)).toEqual(["paper-z", "paper-a"]);
  });

  it("uses canonical paper ID as the final deterministic tie-break", () => {
    const paperA = paper("paper-a", 0.8);
    const paperZ = paper("paper-z", 0.8);

    expect(comparePaperDisplayRank(
      { paper: paperZ, queryRelevance: 0.9, originalRank: 2 },
      { paper: paperA, queryRelevance: 0.9, originalRank: 2 },
    )).toBeGreaterThan(0);
  });

  it("places a paper missing graph semantics after scored papers", () => {
    const searchOrder = [paper("missing", 0.9), paper("scored", 0.2)];

    const displayed = orderPapersByQueryRelevance(searchOrder, [node("scored", 0.1)]);

    expect(displayed.map(({ id }) => id)).toEqual(["scored", "missing"]);
  });
});
