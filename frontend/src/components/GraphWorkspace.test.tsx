import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  acquireAndParsePDF,
  extractPaperInsights,
  getCachedPaperAnalysis,
  getLLMProviders,
} from "../api/client";
import type {
  CachedPaperAnalysisResponse,
  GraphResponse,
  InsightsResponse,
  Paper,
  ParsedPaper,
} from "../types/paper";
import { edgeWidth } from "../utils/graph";
import { GraphWorkspace } from "./GraphWorkspace";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    acquireAndParsePDF: vi.fn(),
    extractPaperInsights: vi.fn(),
    getCachedPaperAnalysis: vi.fn(),
    getLLMProviders: vi.fn(),
  };
});

const papers: Paper[] = [
  {
    id: "paper-a", title: "Agentic Networks", abstract: "Network agent research.",
    authors: ["Ada Author"], publication_year: 2026, publication_date: null,
    venue: "Systems", doi: null, arxiv_id: null, openalex_id: null,
    semantic_scholar_id: null, url: null, pdf_url: null, citation_count: 4,
    source_names: ["openalex"], semantic_score: 0.9, reranker_score: 4.1,
  },
  {
    id: "paper-b", title: "Secure Agents", abstract: "Security for autonomous agents.",
    authors: ["Ben Scholar"], publication_year: 2025, publication_date: null,
    venue: null, doi: null, arxiv_id: null, openalex_id: null,
    semantic_scholar_id: null, url: null, pdf_url: null, citation_count: 2,
    source_names: ["arxiv"], semantic_score: 0.7, reranker_score: 3.4,
  },
  {
    id: "paper-c", title: "Fern Morphology", abstract: null,
    authors: [], publication_year: 2020, publication_date: null, venue: null,
    doi: null, arxiv_id: null, openalex_id: null, semantic_scholar_id: null,
    url: null, pdf_url: null, citation_count: 1, source_names: ["openalex"],
    semantic_score: 0.2, reranker_score: -1,
  },
];

const graph: GraphResponse = {
  semantics_version: "m3.4-m3.6-v1",
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
  nodes: [
    { paper_id: "paper-a", title: "Agentic Networks", query_relevance: 0.9, node_weight: 0.9, node_radius: 26, information_completeness: 1, node_opacity: 1 },
    { paper_id: "paper-b", title: "Secure Agents", query_relevance: 0.7, node_weight: 0.7, node_radius: 22, information_completeness: 0.78, node_opacity: 0.78 },
    { paper_id: "paper-c", title: "Fern Morphology", query_relevance: 0.2, node_weight: 0.2, node_radius: 15, information_completeness: 0.4, node_opacity: 0.4 },
  ],
  edges: [
    { source_paper_id: "paper-a", target_paper_id: "paper-b", paper_similarity: 0.8, edge_weight: 0.8, selection_reason: "both_top_k" },
    { source_paper_id: "paper-a", target_paper_id: "paper-c", paper_similarity: 0.6, edge_weight: 0.6, selection_reason: "source_top_k" },
    { source_paper_id: "paper-b", target_paper_id: "paper-c", paper_similarity: 0.5, edge_weight: 0.5, selection_reason: "target_top_k" },
  ],
};

function parsedDocument(paperId: string, title: string): ParsedPaper {
  return {
    paper_id: paperId, title, authors: ["Ada Author"], abstract: "Parsed abstract.",
    abstract_chunks: [], sections: [], references: [], parser: "grobid",
    parser_version: "0.9.1",
    source_pdf: {
      acquisition_method: "upload", url: null, sha256: paperId.padEnd(64, "a").slice(0, 64),
      size_bytes: 100,
    },
  };
}

function insightResponse(paperId: string, cached = false): InsightsResponse {
  return {
    paper_id: paperId, document_fingerprint: "b".repeat(64), model: "qwen3:1.7b",
    extraction_version: "m2-final-v7-closeout", cached,
    insights: {
      paper_overview: [{
        claim: `Grounded overview for ${paperId}.`,
        evidence: [{ chunk_id: "chunk-1", quote: "Grounded source quote." }],
      }],
      research_problem: [], methods: [], key_contributions: [], evaluation: [],
      main_findings: [], why_it_matters: [], target_audience: [], limitations: [], future_work: [],
    },
  };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getLLMProviders).mockResolvedValue({
    default_provider: "ollama",
    providers: [{ provider_id: "ollama", model: "qwen3:1.7b", configured: true, cloud: false }],
  });
  vi.mocked(getCachedPaperAnalysis).mockResolvedValue({
    paper_id: "paper-a",
    insight_provider: "ollama",
    document: {
      paper_id: "paper-a", title: "Agentic Networks", authors: ["Ada Author"],
      abstract: "Network agent research.", abstract_chunks: [], sections: [], references: [],
      parser: "grobid", parser_version: "0.9.1",
      source_pdf: { acquisition_method: "upload", url: null, sha256: "a".repeat(64), size_bytes: 100 },
    },
    insights: {
      paper_id: "paper-a", document_fingerprint: "b".repeat(64), model: "qwen3:1.7b",
      extraction_version: "m2-final-v7-closeout", cached: true,
      insights: {
        paper_overview: [{ claim: "A complete researcher-level overview that ends with a complete conclusion.", evidence: [{ chunk_id: "chunk-1", quote: "Grounded source quote." }] }],
        research_problem: [], methods: [], key_contributions: [], evaluation: [],
        main_findings: [], why_it_matters: [], target_audience: [], limitations: [], future_work: [],
      },
    },
  });
});

describe("GraphWorkspace", () => {
  it("renders backend node and edge semantics with the complete paper list", async () => {
    const { container } = render(
      <GraphWorkspace papers={papers} graph={graph} graphLoading={false} graphError={null} />,
    );

    await waitFor(() => expect(container.querySelectorAll(".graph-node")).toHaveLength(3));
    expect(container.querySelectorAll("[data-paper-row-id]")).toHaveLength(3);
    const firstNode = container.querySelector<SVGCircleElement>('[data-paper-id="paper-a"]')!;
    expect(firstNode).toHaveAttribute("r", "26");
    expect(firstNode).toHaveAttribute("opacity", "1");
    expect(firstNode).toHaveAttribute("data-color-token", "rw-periwinkle");
    const edge = container.querySelector<SVGLineElement>(".graph-edge")!;
    expect(edge).toHaveAttribute("data-color-token", "rw-edge");
    expect(Number(edge.getAttribute("stroke-width"))).toBe(edgeWidth(0.8));
    expect(edge).toHaveAttribute("stroke-opacity", "1");
    expect(edgeWidth(0.92)).toBeGreaterThan(edgeWidth(0.8));
    expect(edgeWidth(0.8)).toBeGreaterThan(edgeWidth(0.65));
    expect(edgeWidth(0.65)).toBeGreaterThan(edgeWidth(0.35));
    expect(edgeWidth(0.92) - edgeWidth(0.65)).toBeGreaterThan(3);
    const legendThin = container.querySelector<HTMLElement>(".edge-thin")!;
    const legendThick = container.querySelector<HTMLElement>(".edge-thick")!;
    expect(legendThin.style.height).toBe(`${edgeWidth(0.35)}px`);
    expect(legendThick.style.height).toBe(`${edgeWidth(1)}px`);
  });

  it("keeps node hover semantics and links the corresponding paper row", async () => {
    const { container } = render(
      <GraphWorkspace papers={papers} graph={graph} graphLoading={false} graphError={null} />,
    );
    await waitFor(() => expect(container.querySelectorAll(".graph-node")).toHaveLength(3));
    const nodeA = container.querySelector<SVGCircleElement>('[data-paper-id="paper-a"]')!;
    const nodeB = container.querySelector<SVGCircleElement>('[data-paper-id="paper-b"]')!;
    const rowA = container.querySelector<HTMLElement>('[data-paper-row-id="paper-a"]')!;
    const rowB = container.querySelector<HTMLElement>('[data-paper-row-id="paper-b"]')!;
    const edgeAB = container.querySelector<SVGLineElement>(
      '[data-source-id="paper-a"][data-target-id="paper-b"]',
    )!;
    const edgeAC = container.querySelector<SVGLineElement>(
      '[data-source-id="paper-a"][data-target-id="paper-c"]',
    )!;

    fireEvent.mouseEnter(nodeA);
    expect(rowA).toHaveClass("is-active");
    expect(edgeAB).toHaveClass("is-active");
    expect(edgeAC).toHaveClass("is-active");
    expect(screen.getByRole("tooltip")).toHaveTextContent("Query relevance: 0.90");
    fireEvent.mouseLeave(nodeA);

    fireEvent.mouseEnter(rowB);
    expect(nodeB).toHaveClass("is-active");
    fireEvent.mouseLeave(rowB);
  });

  it("highlights only the exact hovered edge, its endpoints, and their two rows", async () => {
    const originalGraph = JSON.parse(JSON.stringify(graph)) as GraphResponse;
    const { container } = render(
      <GraphWorkspace papers={papers} graph={graph} graphLoading={false} graphError={null} />,
    );
    await waitFor(() => expect(container.querySelectorAll(".graph-node")).toHaveLength(3));
    const nodeA = container.querySelector<SVGCircleElement>('[data-paper-id="paper-a"]')!;
    const nodeB = container.querySelector<SVGCircleElement>('[data-paper-id="paper-b"]')!;
    const nodeC = container.querySelector<SVGCircleElement>('[data-paper-id="paper-c"]')!;
    const rowA = container.querySelector<HTMLElement>('[data-paper-row-id="paper-a"]')!;
    const rowB = container.querySelector<HTMLElement>('[data-paper-row-id="paper-b"]')!;
    const rowC = container.querySelector<HTMLElement>('[data-paper-row-id="paper-c"]')!;
    const edgeAB = container.querySelector<SVGLineElement>(
      '[data-source-id="paper-a"][data-target-id="paper-b"]',
    )!;
    const edgeAC = container.querySelector<SVGLineElement>(
      '[data-source-id="paper-a"][data-target-id="paper-c"]',
    )!;
    const edgeBC = container.querySelector<SVGLineElement>(
      '[data-source-id="paper-b"][data-target-id="paper-c"]',
    )!;
    const originalWidth = edgeAB.getAttribute("stroke-width");

    fireEvent.mouseEnter(edgeAB);

    expect(edgeAB).toHaveClass("is-active", "is-hovered-edge");
    expect(edgeAB).not.toHaveClass("is-muted");
    expect(edgeAC).not.toHaveClass("is-active", "is-hovered-edge");
    expect(edgeBC).not.toHaveClass("is-active", "is-hovered-edge");
    expect(edgeAC).toHaveClass("is-muted");
    expect(edgeBC).toHaveClass("is-muted");
    expect(nodeA).toHaveClass("is-active", "is-edge-endpoint");
    expect(nodeB).toHaveClass("is-active", "is-edge-endpoint");
    expect(nodeC).not.toHaveClass("is-active", "is-edge-endpoint");
    expect(nodeC).toHaveClass("is-muted");
    expect(rowA).toHaveClass("is-active", "is-edge-endpoint");
    expect(rowB).toHaveClass("is-active", "is-edge-endpoint");
    expect(rowA).toHaveAttribute("data-highlight-token", "rw-periwinkle-light");
    expect(rowB).toHaveAttribute("data-highlight-token", "rw-periwinkle-light");
    expect(rowC).not.toHaveClass("is-active", "is-edge-endpoint");
    expect(screen.getByRole("tooltip")).toHaveTextContent("Semantic similarity: 0.80");
    expect(nodeA).toHaveAttribute("r", "26");
    expect(nodeA).toHaveAttribute("opacity", "1");
    expect(nodeB).toHaveAttribute("r", "22");
    expect(nodeB).toHaveAttribute("opacity", "0.78");
    expect(edgeAB).toHaveAttribute("stroke-width", originalWidth);
    expect(edgeAB).toHaveAttribute("stroke-opacity", "1");
    expect(graph).toEqual(originalGraph);

    fireEvent.mouseLeave(edgeAB);

    expect(container.querySelectorAll(".is-edge-endpoint")).toHaveLength(0);
    expect(container.querySelectorAll(".graph-edge.is-active")).toHaveLength(0);
    expect(container.querySelectorAll(".graph-edge.is-muted")).toHaveLength(0);
    expect(container.querySelectorAll(".graph-node.is-muted")).toHaveLength(0);
    expect(container.querySelectorAll(".paper-list-row.is-active")).toHaveLength(0);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("preserves persistent selection across exact-edge hover transitions", async () => {
    const { container } = render(
      <GraphWorkspace papers={papers} graph={graph} graphLoading={false} graphError={null} />,
    );
    await waitFor(() => expect(container.querySelectorAll(".graph-node")).toHaveLength(3));
    const nodeC = container.querySelector<SVGCircleElement>('[data-paper-id="paper-c"]')!;
    const edgeAB = container.querySelector<SVGLineElement>(
      '[data-source-id="paper-a"][data-target-id="paper-b"]',
    )!;

    fireEvent.click(nodeC);
    expect(nodeC).toHaveClass("is-selected");
    fireEvent.mouseEnter(edgeAB);
    expect(nodeC).toHaveClass("is-selected", "is-muted");
    fireEvent.mouseLeave(edgeAB);
    expect(nodeC).toHaveClass("is-selected");
    expect(nodeC).not.toHaveClass("is-muted");
  });

  it("opens the same detail experience, preserves the graph, and restores the list", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <GraphWorkspace papers={papers} graph={graph} graphLoading={false} graphError={null} />,
    );
    await waitFor(() => expect(container.querySelectorAll(".graph-node")).toHaveLength(3));

    fireEvent.click(container.querySelector('[data-paper-id="paper-a"]')!);
    expect(screen.getByTestId("research-graph")).toBeInTheDocument();
    expect(await screen.findByRole("complementary", { name: /details for agentic networks/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /back/i }))
      .toHaveClass("back-button");
    expect(screen.getByRole("button", { name: /back/i }))
      .toHaveAttribute("data-design-token", "rw-periwinkle-lighter");
    const overview = await screen.findByText("A complete researcher-level overview that ends with a complete conclusion.");
    expect(overview.textContent?.endsWith("...")).toBe(false);
    expect(overview).toHaveClass("paper-overview-text");
    await user.click(screen.getByText("Evidence (1)"));
    expect(screen.getByText(/Grounded source quote/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /back/i }));
    expect(container.querySelectorAll("[data-paper-row-id]")).toHaveLength(3);

    await user.click(container.querySelector('[data-paper-row-id="paper-a"]')!);
    expect(await screen.findByRole("complementary", { name: /details for agentic networks/i })).toBeInTheDocument();
  });

  it("keeps paper results usable when graph generation fails", () => {
    const { container } = render(
      <GraphWorkspace papers={papers} graph={null} graphLoading={false} graphError="Model unavailable" />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Papers remain available");
    expect(container.querySelectorAll("[data-paper-row-id]")).toHaveLength(3);
  });

  it("keeps backend-authoritative paper state synchronized across a long-lived session", async () => {
    const user = userEvent.setup();
    const states = new Map<string, CachedPaperAnalysisResponse>();
    vi.mocked(getCachedPaperAnalysis).mockImplementation(async (paperId) =>
      states.get(paperId) ?? { paper_id: paperId, document: null, insights: null, insight_provider: null },
    );
    vi.mocked(acquireAndParsePDF).mockImplementation(async (paper) => {
      const document = parsedDocument(paper.id, paper.title);
      states.set(paper.id, {
        paper_id: paper.id, document, insights: null, insight_provider: null,
      });
      return {
        status: "upload", paper_id: paper.id, document,
        message: "PDF parsed.",
      };
    });
    vi.mocked(extractPaperInsights).mockImplementation(async (document) => {
      const insights = insightResponse(document.paper_id);
      states.set(document.paper_id, {
        paper_id: document.paper_id, document, insights: { ...insights, cached: true },
        insight_provider: "ollama",
      });
      return insights;
    });
    const { container } = render(
      <GraphWorkspace papers={papers} graph={graph} graphLoading={false} graphError={null} />,
    );

    await user.click(container.querySelector('[data-paper-row-id="paper-a"]')!);
    await user.click(await screen.findByRole("button", { name: /get pdf/i }));
    expect(await screen.findByText("Parsed abstract.")).toBeInTheDocument();
    expect(acquireAndParsePDF).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /extract grounded insights/i }));
    expect(await screen.findByText("Grounded overview for paper-a.")).toBeInTheDocument();
    expect(extractPaperInsights).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /back/i }));
    expect(container.querySelectorAll("[data-paper-row-id]")).toHaveLength(3);
    await user.click(container.querySelector('[data-paper-row-id="paper-b"]')!);
    expect(await screen.findByRole("button", { name: /get pdf/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /back/i }));
    await user.click(container.querySelector('[data-paper-row-id="paper-a"]')!);
    expect(await screen.findByText("Grounded overview for paper-a.")).toBeInTheDocument();
    expect(extractPaperInsights).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /extract grounded insights/i }));
    await waitFor(() => expect(extractPaperInsights).toHaveBeenCalledTimes(2));
    expect(container.querySelectorAll(".graph-node")).toHaveLength(3);
  });

  it("allows a failed extraction to be retried without stale failure state", async () => {
    const user = userEvent.setup();
    const document = parsedDocument("paper-a", "Agentic Networks");
    const valid = insightResponse("paper-a");
    let cachedInsights: InsightsResponse | null = null;
    vi.mocked(getCachedPaperAnalysis).mockImplementation(async () => ({
      paper_id: "paper-a", document, insights: cachedInsights, insight_provider: "ollama",
    }));
    vi.mocked(extractPaperInsights)
      .mockRejectedValueOnce(new Error("Invalid provider output"))
      .mockImplementationOnce(async () => {
        cachedInsights = { ...valid, cached: true };
        return valid;
      });
    const { container } = render(
      <GraphWorkspace papers={papers} graph={graph} graphLoading={false} graphError={null} />,
    );

    await user.click(container.querySelector('[data-paper-row-id="paper-a"]')!);
    const extract = await screen.findByRole("button", { name: /extract grounded insights/i });
    await user.click(extract);
    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid provider output");
    await user.click(extract);
    expect(await screen.findByText("Grounded overview for paper-a.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(extractPaperInsights).toHaveBeenCalledTimes(2);
  });
});
