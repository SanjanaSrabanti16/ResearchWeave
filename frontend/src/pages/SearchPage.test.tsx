import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GraphResponse, SearchResponse } from "../types/paper";
import { SearchPage } from "./SearchPage";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

const successfulResponse: SearchResponse = {
  query: "visual agents",
  overall_status: "success",
  candidate_count: 24,
  deduplicated_count: 20,
  ranked_count: 1,
  provider_status: {
    openalex: {
      status: "ok", successful_requests: 4, failed_requests: 0,
      cached_requests: 0, message: null,
    },
    semantic_scholar: {
      status: "degraded", successful_requests: 3, failed_requests: 1,
      cached_requests: 0,
      message: "Some query variants failed; partial results remain available.",
    },
    arxiv: {
      status: "ok", successful_requests: 4, failed_requests: 0,
      cached_requests: 0, message: null,
    },
  },
  warnings: [],
  papers: [
    {
      id: "paper-1",
      title: "Agentic Visualization",
      abstract: "A relevant abstract.",
      authors: ["Ada Author"],
      publication_year: 2025,
      publication_date: "2025-01-01",
      venue: "VIS",
      doi: "10.1/example",
      arxiv_id: null,
      openalex_id: "W1",
      semantic_scholar_id: null,
      url: "https://example.test/paper",
      pdf_url: null,
      citation_count: 3,
      source_names: ["openalex"],
      semantic_score: 0.75,
      reranker_score: 4.2,
    },
  ],
};

const successfulGraph: GraphResponse = {
  semantics_version: "m3.4-m3.6-v4",
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
  nodes: [{
    paper_id: "paper-1", title: "Agentic Visualization", query_relevance: 0.75,
    node_weight: 0.75, node_radius: 23, information_completeness: 0.55,
    node_opacity: 0.685,
  }],
  edges: [],
  related_concepts: [
    { text: "Agentic visualization", query_similarity: 0.94 },
    { text: "Visual agents", query_similarity: 0.91 },
    { text: "Interactive systems", query_similarity: 0.82 },
    { text: "Autonomous workflows", query_similarity: 0.78 },
    { text: "Visual analytics", query_similarity: 0.75 },
  ],
};

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

beforeEach(() => fetchMock.mockReset());

describe("SearchPage", () => {
  it("renders the search form", () => {
    render(<SearchPage />);
    const logo = screen.getByRole("img", { name: "ResearchWeave" });
    expect(logo).toHaveClass("brand-logo");
    expect(logo).toHaveAttribute("src", expect.stringContaining("researchWeave_Logo.png"));
    expect(document.querySelector(".brand-block h1")).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: /research topic/i })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: /start year/i })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: /end year/i })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: /papers/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^search$/i }))
      .toHaveClass("search-button");
    expect(screen.getByRole("button", { name: /^search$/i }))
      .toHaveAttribute("data-design-token", "rw-periwinkle");
  });

  it("shows related concepts and compact source diagnostics after search", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(successfulResponse))
      .mockResolvedValueOnce(jsonResponse(successfulGraph));
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.click(screen.getByRole("button", { name: /^search$/i }));

    await screen.findByTestId("research-graph");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/search");
    expect(options).toMatchObject({ method: "POST", signal: expect.any(AbortSignal) });
    expect(JSON.parse(options.body)).toEqual({ query: "visual agents", limit: 20 });
    expect(fetchMock.mock.calls[1][0]).toBe("http://localhost:8000/api/graph");
    expect(await screen.findByText("Agentic Visualization")).toBeInTheDocument();
    expect(screen.getByText("Related to your query")).toBeInTheDocument();
    expect(screen.getByText("Visual agents")).toHaveClass("related-concept-chip");
    expect(screen.getAllByText(/visual/i).filter((element) =>
      element.classList.contains("related-concept-chip"))).toHaveLength(3);
    expect(screen.getByLabelText("3 active scholarly sources")).toHaveTextContent(
      "Sources · 3 active",
    );
    expect(document.querySelectorAll(".provider")).toHaveLength(0);
    expect(screen.getByText(/partial results remain available/i)).toBeInTheDocument();
    expect(document.querySelectorAll(".search-output > .status.warning")).toHaveLength(0);
    expect(screen.queryByText(/semantic scholar is temporarily unavailable/i)).not.toBeInTheDocument();
  });

  it("keeps partial provider outages compact and exposes expandable details", async () => {
    const partial = {
      ...successfulResponse,
      provider_status: {
        ...successfulResponse.provider_status,
        arxiv: {
          status: "unavailable" as const,
          successful_requests: 0,
          failed_requests: 1,
          cached_requests: 0,
          message: "No requests succeeded; this provider is temporarily unavailable.",
        },
      },
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse(partial))
      .mockResolvedValueOnce(jsonResponse(successfulGraph));
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.click(screen.getByRole("button", { name: /^search$/i }));
    await screen.findByTestId("research-graph");

    const summary = screen.getByLabelText("2 active scholarly sources");
    expect(summary).toHaveTextContent("1 unavailable");
    expect(document.querySelectorAll(".search-output > .status.warning")).toHaveLength(0);
    await user.click(summary);
    expect(screen.getByText(/arxiv/i, { selector: ".source-status-details strong" }))
      .toBeInTheDocument();
  });

  it("reorders the display after graph semantics without rerunning scholarly search", async () => {
    const lowerM1Paper = {
      ...successfulResponse.papers[0],
      id: "paper-2",
      title: "Higher Query Relevance",
      semantic_score: 0.3,
      reranker_score: 1.1,
    };
    const searchResponse = {
      ...successfulResponse,
      ranked_count: 2,
      papers: [successfulResponse.papers[0], lowerM1Paper],
    };
    const graphResponse: GraphResponse = {
      ...successfulGraph,
      nodes: [
        { ...successfulGraph.nodes[0], query_relevance: 0.6 },
        {
          ...successfulGraph.nodes[0],
          paper_id: "paper-2",
          title: "Higher Query Relevance",
          query_relevance: 0.95,
          node_weight: 0.95,
          node_radius: 27,
        },
      ],
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse(searchResponse))
      .mockResolvedValueOnce(jsonResponse(graphResponse));
    const user = userEvent.setup();
    const { container } = render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.click(screen.getByRole("button", { name: /^search$/i }));

    await screen.findByTestId("research-graph");
    const rows = Array.from(container.querySelectorAll<HTMLElement>("[data-paper-row-id]"));
    expect(rows.map((row) => row.dataset.paperRowId)).toEqual(["paper-2", "paper-1"]);
    expect(rows[0].querySelector(".paper-row-rank")).toHaveTextContent("1");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/api/search")))
      .toHaveLength(1);
  });

  it("shows loading while a search is pending", async () => {
    let resolveSearch!: (response: Response) => void;
    fetchMock.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveSearch = resolve;
      }),
    );
    fetchMock.mockResolvedValueOnce(jsonResponse(successfulGraph));
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.click(screen.getByRole("button", { name: /^search$/i }));
    expect(screen.getByRole("status")).toHaveTextContent(/searching papers/i);

    resolveSearch(jsonResponse(successfulResponse));
    expect(await screen.findByText("Agentic Visualization")).toBeInTheDocument();
  });

  it("renders an API failure", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ detail: "Provider service unavailable" }, 503),
    );
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.click(screen.getByRole("button", { name: /^search$/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Provider service unavailable");
  });

  it("rejects an invalid year range before calling the API", async () => {
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.type(screen.getByRole("spinbutton", { name: /start year/i }), "2026");
    await user.type(screen.getByRole("spinbutton", { name: /end year/i }), "2020");
    await user.click(screen.getByRole("button", { name: /^search$/i }));

    expect(screen.getByRole("alert")).toHaveTextContent(/start year must be before/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
