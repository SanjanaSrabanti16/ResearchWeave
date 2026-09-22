import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { acquireAndParsePDF, APIError, extractPaperInsights, uploadAndParsePDF } from "../api/client";
import type { InsightsResponse, Paper, ParsedPaper } from "../types/paper";
import { PaperPdfPanel } from "./PaperPdfPanel";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    acquireAndParsePDF: vi.fn(),
    uploadAndParsePDF: vi.fn(),
    extractPaperInsights: vi.fn(),
  };
});

const paper: Paper = {
  id: "paper-1",
  title: "Search result title",
  abstract: null,
  authors: [],
  publication_year: 2024,
  publication_date: null,
  venue: null,
  doi: "10.1000/example",
  arxiv_id: null,
  openalex_id: null,
  semantic_scholar_id: null,
  url: null,
  pdf_url: null,
  citation_count: null,
  source_names: ["openalex"],
  semantic_score: null,
  reranker_score: null,
};

const document: ParsedPaper = {
  paper_id: "paper-1",
  title: "Parsed title",
  authors: ["Ada Scholar"],
  abstract: "Parsed abstract.",
  abstract_chunks: [],
  sections: [
    {
      id: "section-0001",
      heading: "Introduction",
      level: 1,
      text: "Section evidence.",
      page_start: null,
      page_end: null,
      chunks: [
        {
          id: "chunk-1",
          section_id: "section-0001",
          text: "Section evidence.",
          start_char: 0,
          end_char: 17,
          page_start: null,
          page_end: null,
        },
      ],
    },
  ],
  references: [{}],
  parser: "grobid",
  parser_version: "0.9.0-crf",
  source_pdf: {
    acquisition_method: "upload",
    url: null,
    sha256: "a".repeat(64),
    size_bytes: 100,
  },
};

describe("PaperPdfPanel", () => {
  beforeEach(() => {
    vi.mocked(acquireAndParsePDF).mockReset();
    vi.mocked(uploadAndParsePDF).mockReset();
    vi.mocked(extractPaperInsights).mockReset();
  });

  it("shows upload fallback and previews a parsed upload", async () => {
    vi.mocked(acquireAndParsePDF).mockResolvedValue({
      status: "upload_required",
      paper_id: "paper-1",
      document: null,
      message: "Upload required.",
    });
    vi.mocked(uploadAndParsePDF).mockResolvedValue({
      status: "upload",
      paper_id: "paper-1",
      document,
      message: "Parsed successfully.",
    });
    const user = userEvent.setup();
    render(<PaperPdfPanel paper={paper} />);

    await user.click(screen.getByRole("button", { name: "Get PDF" }));
    const input = await screen.findByLabelText("Upload a PDF you are authorized to use");
    await user.upload(input, new File(["%PDF-1.7"], "paper.pdf", { type: "application/pdf" }));
    await user.click(screen.getByRole("button", { name: "Upload and parse" }));

    expect(await screen.findByText("Parsed title")).toBeInTheDocument();
    expect(screen.getByText("Parsed abstract.")).toBeInTheDocument();
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("References extracted: 1")).toBeInTheDocument();
    expect(uploadAndParsePDF).toHaveBeenCalledOnce();
  });

  it("previews a successfully acquired open-access PDF", async () => {
    vi.mocked(acquireAndParsePDF).mockResolvedValue({
      status: "unpaywall",
      paper_id: "paper-1",
      document: {
        ...document,
        source_pdf: { ...document.source_pdf, acquisition_method: "unpaywall" },
      },
      message: "Open-access PDF parsed.",
    });
    const user = userEvent.setup();
    render(<PaperPdfPanel paper={paper} />);

    await user.click(screen.getByRole("button", { name: "Get PDF" }));

    expect(await screen.findByText("Open-access PDF parsed.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Get PDF" })).not.toBeInTheDocument();
  });

  it("reports parser unavailability without presenting upload as a remedy", async () => {
    vi.mocked(acquireAndParsePDF).mockRejectedValue(
      new APIError("GROBID parser is unavailable", 503),
    );
    const user = userEvent.setup();
    render(<PaperPdfPanel paper={paper} />);

    await user.click(screen.getByRole("button", { name: "Get PDF" }));

    expect(await screen.findByText("GROBID parser is unavailable")).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Upload a PDF you are authorized to use"),
    ).not.toBeInTheDocument();
  });

  it("renders all insight fields and expandable validated evidence", async () => {
    vi.mocked(acquireAndParsePDF).mockResolvedValue({
      status: "arxiv",
      paper_id: "paper-1",
      document,
      message: "Parsed successfully.",
    });
    const insightResponse: InsightsResponse = {
      paper_id: "paper-1",
      document_fingerprint: "b".repeat(64),
      model: "qwen3:4b",
      extraction_version: "m2b-v1",
      cached: false,
      insights: {
        research_problem: [],
        methods: [{ claim: "The paper uses a method.", evidence: [{ chunk_id: "chunk-1", quote: "Section evidence." }] }],
        key_contributions: [],
        main_findings: [],
        why_it_matters: [],
        target_audience: [],
        limitations: [],
        future_work: [],
      },
    };
    vi.mocked(extractPaperInsights).mockResolvedValue(insightResponse);
    const user = userEvent.setup();
    render(<PaperPdfPanel paper={paper} />);

    await user.click(screen.getByRole("button", { name: "Get PDF" }));
    await user.click(await screen.findByRole("button", { name: "Extract grounded insights" }));

    expect(await screen.findByText("The paper uses a method.")).toBeInTheDocument();
    for (const heading of ["Problem", "Methods", "Contributions", "Findings", "Why It Matters", "Audience", "Limitations", "Future Work"]) {
      expect(screen.getByRole("heading", { name: heading })).toBeInTheDocument();
    }
    await user.click(screen.getByText("Evidence (1)"));
    expect(screen.getByText("“Section evidence.”")).toBeInTheDocument();
    expect(screen.getByText("chunk-1")).toBeInTheDocument();
    expect(extractPaperInsights).toHaveBeenCalledWith(document, expect.any(Function));
  });

  it("keeps parsed metadata visible while showing extraction progress", async () => {
    vi.mocked(acquireAndParsePDF).mockResolvedValue({
      status: "arxiv",
      paper_id: "paper-1",
      document,
      message: "Parsed successfully.",
    });
    let finish!: (value: InsightsResponse) => void;
    vi.mocked(extractPaperInsights).mockImplementation((_, onProgress) => {
      onProgress?.("Generating grounded insights");
      return new Promise((resolve) => { finish = resolve; });
    });
    const user = userEvent.setup();
    render(<PaperPdfPanel paper={paper} />);

    await user.click(screen.getByRole("button", { name: "Get PDF" }));
    await user.click(await screen.findByRole("button", { name: "Extract grounded insights" }));

    expect(screen.getByText("Parsed title")).toBeInTheDocument();
    expect(screen.getByText("Parsed abstract.")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Generating grounded insights");
    finish({
      paper_id: "paper-1",
      document_fingerprint: "b".repeat(64),
      model: "qwen3:4b",
      extraction_version: "m2b-v5-selective",
      cached: false,
      insights: {
        research_problem: [], methods: [], key_contributions: [], main_findings: [],
        why_it_matters: [], target_audience: [], limitations: [], future_work: [],
      },
    });
    expect(await screen.findByText(/Grounded with qwen3:4b/)).toBeInTheDocument();
  });

  it("reports local Ollama failure without showing unsupported insights", async () => {
    vi.mocked(acquireAndParsePDF).mockResolvedValue({
      status: "arxiv",
      paper_id: "paper-1",
      document,
      message: "Parsed successfully.",
    });
    vi.mocked(extractPaperInsights).mockRejectedValue(new APIError("Local Ollama is unavailable", 503));
    const user = userEvent.setup();
    render(<PaperPdfPanel paper={paper} />);

    await user.click(screen.getByRole("button", { name: "Get PDF" }));
    await user.click(await screen.findByRole("button", { name: "Extract grounded insights" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Local Ollama is unavailable");
    expect(screen.queryByText("Research insights")).not.toBeInTheDocument();
  });
});
