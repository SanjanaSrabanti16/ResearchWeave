import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GraphResponse, SearchResponse } from "../types/paper";
import { SearchPage } from "./SearchPage";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

const successfulResponse: SearchResponse = {
  query: "visual agents",
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
  semantics_version: "m3.4-m3.6-v1",
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
  nodes: [{
    paper_id: "paper-1", title: "Agentic Visualization", query_relevance: 0.75,
    node_weight: 0.75, node_radius: 23, information_completeness: 0.55,
    node_opacity: 0.55,
  }],
  edges: [],
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
    expect(screen.getByRole("textbox", { name: /research topic/i })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: /start year/i })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: /end year/i })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: /papers/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^search$/i }))
      .toHaveClass("search-button");
    expect(screen.getByRole("button", { name: /^search$/i }))
      .toHaveAttribute("data-design-token", "rw-periwinkle");
  });

  it("submits the API request and renders one authoritative provider-health state", async () => {
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
    expect(screen.getByText(/semantic scholar: degraded/i)).toBeInTheDocument();
    expect(screen.getByText(/partial results remain available/i)).toBeInTheDocument();
    expect(screen.queryByText(/semantic scholar: ok/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/semantic scholar is temporarily unavailable/i)).not.toBeInTheDocument();
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
