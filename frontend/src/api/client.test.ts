import { afterEach, describe, expect, it, vi } from "vitest";

import {
  acquireAndParsePDF,
  extractPaperInsights,
  getPaperRelationship,
  getPaperRelationshipByPair,
  getPaperRelationshipHistory,
  proposePaperRelationship,
  reviewPaperRelationship,
} from "./client";
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
      pmcid: "PMC123456",
      pmcids: ["PMC123456"],
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
      pmcid: "PMC123456",
      pmcids: ["PMC123456"],
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

describe("relationship API compatibility", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("proposes from canonical IDs and optional constrained types", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await proposePaperRelationship({
      source_paper_id: "paper-a",
      target_paper_id: "paper-b",
      requested_relationship_types: ["shared_method"],
    });

    expect(fetchMock.mock.calls[0][0]).toMatch(/\/api\/relationships\/propose$/);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body as string)).toEqual({
      source_paper_id: "paper-a",
      target_paper_id: "paper-b",
      requested_relationship_types: ["shared_method"],
    });
  });

  it("submits only reviewable human fields", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await reviewPaperRelationship("proposal/1", {
      decision: "edited",
      edited_relationship_types: ["related_application"],
      edited_summary: "Related application contexts.",
      reviewer_note: "Reviewed",
    });

    expect(fetchMock.mock.calls[0][0]).toMatch(/\/proposal%2F1\/review$/);
    expect(JSON.parse(fetchMock.mock.calls[0][1].body as string)).not.toHaveProperty(
      "source_evidence_ids",
    );
  });

  it("retrieves encoded proposal and history paths", async () => {
    const fetchMock = vi.fn().mockImplementation(async () =>
      new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await getPaperRelationship("proposal/1");
    await getPaperRelationshipHistory("proposal/1");

    expect(fetchMock.mock.calls[0][0]).toMatch(/\/api\/relationships\/proposal%2F1$/);
    expect(fetchMock.mock.calls[1][0]).toMatch(/\/api\/relationships\/proposal%2F1\/history$/);
  });

  it("performs a no-store cache lookup by encoded canonical pair", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response("null", { status: 200, headers: { "Content-Type": "application/json" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await getPaperRelationshipByPair("paper/a", "paper b");

    expect(result).toBeNull();
    expect(fetchMock.mock.calls[0][0]).toContain(
      "/api/relationships/by-pair?source_paper_id=paper%2Fa&target_paper_id=paper+b",
    );
    expect(fetchMock.mock.calls[0][1]).toEqual({ cache: "no-store" });
  });
});
