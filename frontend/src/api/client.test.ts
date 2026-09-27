import { afterEach, describe, expect, it, vi } from "vitest";

import { acquireAndParsePDF, extractPaperInsights } from "./client";
import type { Paper, ParsedPaper } from "../types/paper";

describe("acquireAndParsePDF", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("forwards preserved acquisition metadata to the backend", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: "upload_required",
          paper_id: "paper-1",
          document: null,
          message: "Upload required",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const paper: Paper = {
      id: "paper-1",
      title: "Canonical title",
      abstract: null,
      authors: ["Ada Author"],
      publication_year: 2024,
      publication_date: null,
      venue: "Journal",
      doi: "10.1/example",
      arxiv_id: "2401.12345",
      arxiv_ids: ["2401.12345"],
      openalex_id: "W1",
      semantic_scholar_id: "S1",
      url: "https://publisher.example/article",
      alternate_urls: ["https://arxiv.org/abs/2401.12345"],
      pdf_url: "https://repository.example/paper.pdf",
      alternate_pdf_urls: ["https://arxiv.org/pdf/2401.12345.pdf"],
      citation_count: 1,
      source_names: ["openalex", "arxiv"],
      semantic_score: 0.5,
      reranker_score: 1,
    };

    await acquireAndParsePDF(paper);

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({
      paper_id: "paper-1",
      doi: "10.1/example",
      arxiv_id: "2401.12345",
      arxiv_ids: ["2401.12345"],
      pdf_url: "https://repository.example/paper.pdf",
      alternate_pdf_urls: ["https://arxiv.org/pdf/2401.12345.pdf"],
      url: "https://publisher.example/article",
      alternate_urls: ["https://arxiv.org/abs/2401.12345"],
      title: "Canonical title",
      authors: ["Ada Author"],
      publication_year: 2024,
    });
  });
});

describe("extractPaperInsights", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends the selected provider without any provider secret", async () => {
    const result = {
      paper_id: "paper-1",
      document_fingerprint: "f".repeat(64),
      model: "gemini-3.5-flash",
      extraction_version: "m2-final-v1-evidence-ledger",
      cached: false,
      insights: {
        paper_overview: [], research_problem: [], methods: [], key_contributions: [],
        evaluation: [], main_findings: [], why_it_matters: [], target_audience: [],
        limitations: [], future_work: [],
      },
    };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(`${JSON.stringify({ type: "result", data: result })}\n`, {
        status: 200,
        headers: { "Content-Type": "application/x-ndjson" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const document = {
      paper_id: "paper-1",
      title: "Paper",
      authors: [],
      abstract: null,
      abstract_chunks: [],
      sections: [],
      references: [],
      parser: "grobid",
      parser_version: "0.9.1",
      source_pdf: {
        acquisition_method: "upload",
        url: null,
        sha256: "a".repeat(64),
        size_bytes: 10,
      },
    } satisfies ParsedPaper;

    await extractPaperInsights(document, undefined, "evl_gemma");

    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(body).toEqual({ document, provider: "evl_gemma" });
    expect(JSON.stringify(body)).not.toContain("API_KEY");
  });
});
