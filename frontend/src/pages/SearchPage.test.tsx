import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SearchResponse } from "../types/paper";
import { SearchPage } from "./SearchPage";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

const successfulResponse: SearchResponse = {
  query: "visual agents",
  candidate_count: 24,
  deduplicated_count: 20,
  ranked_count: 1,
  provider_status: { openalex: "ok", semantic_scholar: "error", arxiv: "ok" },
  warnings: ["semantic_scholar is temporarily unavailable"],
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
    expect(screen.getByRole("spinbutton", { name: /results/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /search papers/i })).toBeInTheDocument();
  });

  it("submits the API request and renders papers and provider warnings", async () => {
    fetchMock.mockResolvedValue(jsonResponse(successfulResponse));
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.click(screen.getByRole("button", { name: /search papers/i }));

    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/search");
    expect(options).toMatchObject({ method: "POST", signal: expect.any(AbortSignal) });
    expect(JSON.parse(options.body)).toEqual({ query: "visual agents", limit: 20 });
    expect(await screen.findByText("Agentic Visualization")).toBeInTheDocument();
    expect(screen.getByText("semantic_scholar is temporarily unavailable")).toBeInTheDocument();
  });

  it("shows loading while a search is pending", async () => {
    let resolveSearch!: (response: Response) => void;
    fetchMock.mockReturnValue(
      new Promise((resolve) => {
        resolveSearch = resolve;
      }),
    );
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.click(screen.getByRole("button", { name: /search papers/i }));
    expect(screen.getByRole("status")).toHaveTextContent(/retrieving and ranking/i);

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
    await user.click(screen.getByRole("button", { name: /search papers/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Provider service unavailable");
  });

  it("rejects an invalid year range before calling the API", async () => {
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole("textbox", { name: /research topic/i }), "visual agents");
    await user.type(screen.getByRole("spinbutton", { name: /start year/i }), "2026");
    await user.type(screen.getByRole("spinbutton", { name: /end year/i }), "2020");
    await user.click(screen.getByRole("button", { name: /search papers/i }));

    expect(screen.getByRole("alert")).toHaveTextContent(/start year must be before/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
