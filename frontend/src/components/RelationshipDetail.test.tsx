import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  APIError,
  getPaperRelationshipByPair,
  getPaperRelationshipHistory,
  proposePaperRelationship,
  reviewPaperRelationship,
} from "../api/client";
import type {
  GraphEdge,
  Paper,
  RelationshipProposal,
  RelationshipReview,
  RelationshipReviewHistory,
} from "../types/paper";
import { RelationshipDetail } from "./RelationshipDetail";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    getPaperRelationshipByPair: vi.fn(),
    getPaperRelationshipHistory: vi.fn(),
    proposePaperRelationship: vi.fn(),
    reviewPaperRelationship: vi.fn(),
  };
});

const sourcePaper: Paper = {
  id: "paper-a", title: "Agentic Network Design", abstract: "Agent design.",
  authors: ["Ada"], publication_year: 2026, publication_date: null, venue: null,
  doi: null, arxiv_id: null, openalex_id: null, semantic_scholar_id: null,
  url: null, pdf_url: null, citation_count: 1, source_names: ["arxiv"],
  semantic_score: 0.9, reranker_score: 2,
};

const targetPaper: Paper = {
  ...sourcePaper,
  id: "paper-b",
  title: "Secure Autonomous Systems",
  abstract: "Agent security.",
};

const edge: GraphEdge = {
  source_paper_id: "paper-a",
  target_paper_id: "paper-b",
  paper_similarity: 0.82,
  edge_weight: 0.82,
  selection_reason: "both_top_k",
};

const proposal: RelationshipProposal = {
  proposal_id: "proposal-1",
  source_paper_id: "paper-a",
  target_paper_id: "paper-b",
  semantic_similarity: 0.82,
  relationship_types: ["shared_method"],
  relationships: [
    {
      relationship_type: "shared_method",
      summary: "Both papers use structured agent coordination while preserving distinct security goals.",
      source_evidence_ids: ["a-method"],
      target_evidence_ids: ["b-method"],
      confidence: 0.79,
    },
  ],
  summary: "The complete relationship overview remains fully visible without truncation or ellipsis.",
  evidence_source: [
    {
      paper_id: "paper-a",
      evidence_id: "a-method",
      insight_field: "methods",
      claim: "Paper A coordinates specialized agents.",
      quote: "The framework coordinates specialized agents through a staged workflow.",
      section_id: "a-sec",
      section_heading: "Methods",
      page_start: 4,
      page_end: 4,
    },
  ],
  evidence_target: [
    {
      paper_id: "paper-b",
      evidence_id: "b-method",
      insight_field: "methods",
      claim: "Paper B coordinates security agents.",
      quote: "Security agents coordinate detection and response stages.",
      section_id: "b-sec",
      section_heading: "Methodology",
      page_start: 6,
      page_end: 7,
    },
  ],
  confidence: 0.79,
  source_analysis: {
    document_fingerprint: "a".repeat(64), provider: "ollama", model: "qwen3:1.7b",
    extraction_version: "m2-final-v5-full-document",
  },
  target_analysis: {
    document_fingerprint: "b".repeat(64), provider: "ollama", model: "qwen3:1.7b",
    extraction_version: "m2-final-v5-full-document",
  },
  relationship_provider: "ollama",
  relationship_model: "qwen3:1.7b",
  relationship_pipeline_version: "m4.1-v1",
  created_at: "2026-09-29T12:00:00Z",
  cached: true,
  diagnostics: [],
};

const acceptedReview: RelationshipReview = {
  review_id: "review-1",
  proposal_id: proposal.proposal_id,
  review_version: 1,
  decision: "accepted",
  original_relationship_types: proposal.relationship_types,
  original_summary: proposal.summary,
  relationship_pipeline_version: proposal.relationship_pipeline_version,
  edited_relationship_types: null,
  edited_summary: null,
  reviewer_note: null,
  created_at: "2026-09-29T12:10:00Z",
};

function history(reviews: RelationshipReview[] = []): RelationshipReviewHistory {
  return { proposal_id: proposal.proposal_id, proposal, reviews };
}

function renderDetail(overrides: Partial<React.ComponentProps<typeof RelationshipDetail>> = {}) {
  const props = {
    edge,
    sourcePaper,
    targetPaper,
    onBack: vi.fn(),
    onOpenPaper: vi.fn(),
    ...overrides,
  };
  return { ...render(<RelationshipDetail {...props} />), props };
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getPaperRelationshipByPair).mockResolvedValue(null);
  vi.mocked(getPaperRelationshipHistory).mockResolvedValue(history());
  vi.mocked(proposePaperRelationship).mockResolvedValue(proposal);
  vi.mocked(reviewPaperRelationship).mockResolvedValue(acceptedReview);
});

describe("RelationshipDetail", () => {
  it("loads cache state without generating and shows explicit not-generated action", async () => {
    renderDetail();

    expect(await screen.findByRole("button", { name: /generate relationship analysis/i })).toBeInTheDocument();
    expect(getPaperRelationshipByPair).toHaveBeenCalledWith("paper-a", "paper-b");
    expect(proposePaperRelationship).not.toHaveBeenCalled();
    expect(screen.getByText(/semantic similarity:/i)).toHaveTextContent("0.82");
  });

  it("renders a cached proposal with full summary, readable type, confidence, and bilateral evidence", async () => {
    vi.mocked(getPaperRelationshipByPair).mockResolvedValue(proposal);
    renderDetail();

    expect(await screen.findByText("Shared Method")).toBeInTheDocument();
    const summary = screen.getByText(proposal.relationships[0].summary);
    expect(summary).toHaveTextContent(proposal.relationships[0].summary);
    expect(summary).toHaveClass("relationship-item-summary");
    expect(screen.getByText(proposal.summary)).toHaveTextContent(proposal.summary);
    expect(screen.getByText("Confidence: 0.79")).toBeInTheDocument();

    const sourceEvidence = screen.getByText(/Evidence from Agentic Network Design/).closest("details")!;
    const targetEvidence = screen.getByText(/Evidence from Secure Autonomous Systems/).closest("details")!;
    expect(within(sourceEvidence).getByText(/coordinates specialized agents through/i)).toBeInTheDocument();
    expect(within(sourceEvidence).queryByText(/Security agents coordinate/i)).not.toBeInTheDocument();
    expect(within(targetEvidence).getByText(/Security agents coordinate/i)).toBeInTheDocument();
    expect(within(targetEvidence).queryByText(/specialized agents through/i)).not.toBeInTheDocument();
    expect(within(sourceEvidence).getByText("Section: Methods")).toBeInTheDocument();
    expect(within(targetEvidence).getByText("Pages 6–7")).toBeInTheDocument();
    expect(proposePaperRelationship).not.toHaveBeenCalled();
  });

  it("generates only after an explicit click and blocks duplicate pending calls", async () => {
    let resolveProposal!: (value: RelationshipProposal) => void;
    vi.mocked(proposePaperRelationship).mockReturnValue(
      new Promise((resolve) => {
        resolveProposal = resolve;
      }),
    );
    vi.mocked(getPaperRelationshipByPair)
      .mockResolvedValueOnce(null)
      .mockResolvedValue(proposal);
    renderDetail();
    const generate = await screen.findByRole("button", { name: /generate relationship analysis/i });

    fireEvent.click(generate);
    fireEvent.click(generate);

    expect(await screen.findByRole("status")).toHaveTextContent(/Analyzing evidence from both papers/i);
    expect(proposePaperRelationship).toHaveBeenCalledTimes(1);
    resolveProposal(proposal);
    expect(await screen.findByText("Shared Method")).toBeInTheDocument();
    expect(getPaperRelationshipByPair).toHaveBeenCalledTimes(2);
  });

  it("shows actionable missing-analysis state without starting acquisition or extraction", async () => {
    vi.mocked(getPaperRelationshipByPair).mockRejectedValue(
      new APIError("Paper paper-a has no current evidence-grounded M2 analysis", 409, "CURRENT_ANALYSIS_MISSING"),
    );
    const { props } = renderDetail();

    expect(await screen.findByText(/Paper A needs current structured insights/i)).toBeInTheDocument();
    expect(proposePaperRelationship).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: /open paper a details/i }));
    expect(props.onOpenPaper).toHaveBeenCalledWith("paper-a");
  });

  it("contains generation errors locally and retries safely", async () => {
    vi.mocked(getPaperRelationshipByPair).mockResolvedValueOnce(null).mockResolvedValue(proposal);
    vi.mocked(proposePaperRelationship)
      .mockRejectedValueOnce(new APIError("invalid output", 502, "INVALID_RELATIONSHIP_FORMAT"))
      .mockResolvedValueOnce(proposal);
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole("button", { name: /generate relationship analysis/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be generated/i);
    await user.click(screen.getByRole("button", { name: /retry/i }));
    expect(await screen.findByText("Shared Method")).toBeInTheDocument();
    expect(proposePaperRelationship).toHaveBeenCalledTimes(2);
  });

  it("accepts a proposal and refetches authoritative proposal/history state", async () => {
    vi.mocked(getPaperRelationshipByPair).mockResolvedValue(proposal);
    vi.mocked(getPaperRelationshipHistory)
      .mockResolvedValueOnce(history())
      .mockResolvedValue(history([acceptedReview]));
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole("button", { name: /^accept$/i }));

    expect(reviewPaperRelationship).toHaveBeenCalledWith("proposal-1", { decision: "accepted" });
    expect(
      await screen.findByText("Accepted", { selector: ".relationship-review-status" }),
    ).toBeInTheDocument();
    expect(getPaperRelationshipByPair).toHaveBeenCalledTimes(2);
    expect(getPaperRelationshipHistory).toHaveBeenCalledTimes(2);
  });

  it("submits rejection once and preserves the proposal cards", async () => {
    const rejected = { ...acceptedReview, review_id: "review-r", decision: "rejected" as const };
    vi.mocked(getPaperRelationshipByPair).mockResolvedValue(proposal);
    vi.mocked(getPaperRelationshipHistory)
      .mockResolvedValueOnce(history())
      .mockResolvedValue(history([rejected]));
    const user = userEvent.setup();
    renderDetail();

    await user.click(await screen.findByRole("button", { name: /^reject$/i }));

    expect(reviewPaperRelationship).toHaveBeenCalledWith("proposal-1", { decision: "rejected" });
    expect(
      await screen.findByText("Rejected", { selector: ".relationship-review-status" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Shared Method")).toBeInTheDocument();
  });

  it("edits only relationship types, summary, and optional reviewer note", async () => {
    const editedReview: RelationshipReview = {
      ...acceptedReview,
      review_id: "review-e",
      decision: "edited",
      edited_relationship_types: ["related_application"],
      edited_summary: "The systems apply related coordination patterns.",
      reviewer_note: "Reviewed by a domain expert.",
    };
    vi.mocked(getPaperRelationshipByPair).mockResolvedValue(proposal);
    vi.mocked(getPaperRelationshipHistory)
      .mockResolvedValueOnce(history())
      .mockResolvedValue(history([editedReview]));
    const user = userEvent.setup();
    renderDetail();
    await user.click(await screen.findByRole("button", { name: /^edit$/i }));

    expect(screen.getByLabelText(/relationship summary/i)).toHaveValue(proposal.summary);
    expect(screen.queryByLabelText(/evidence/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/paper id/i)).not.toBeInTheDocument();
    await user.clear(screen.getByLabelText(/relationship summary/i));
    await user.type(
      screen.getByLabelText(/relationship summary/i),
      "The systems apply related coordination patterns.",
    );
    await user.click(screen.getByLabelText("Shared Method"));
    await user.click(screen.getByLabelText("Related Application"));
    await user.type(screen.getByLabelText(/reviewer note/i), "Reviewed by a domain expert.");
    await user.click(screen.getByRole("button", { name: /save edit/i }));

    expect(reviewPaperRelationship).toHaveBeenCalledWith("proposal-1", {
      decision: "edited",
      edited_relationship_types: ["related_application"],
      edited_summary: "The systems apply related coordination patterns.",
      reviewer_note: "Reviewed by a domain expert.",
    });
    expect(await screen.findByText("Human-edited interpretation")).toBeInTheDocument();
    expect(screen.getByText(proposal.summary)).toBeInTheDocument();
  });

  it("renders chronological review history without duplicate frontend entries", async () => {
    const edited = {
      ...acceptedReview,
      review_id: "review-2",
      review_version: 2,
      decision: "edited" as const,
      edited_relationship_types: ["related_application" as const],
      edited_summary: "Edited human interpretation.",
      reviewer_note: "Clarified scope.",
      created_at: "2026-09-29T12:20:00Z",
    };
    vi.mocked(getPaperRelationshipByPair).mockResolvedValue(proposal);
    vi.mocked(getPaperRelationshipHistory).mockResolvedValue(history([acceptedReview, edited]));
    renderDetail();

    expect(await screen.findByText("Review history (2)")).toBeInTheDocument();
    const historyPanel = screen.getByText("Review history (2)").closest("details")!;
    expect(within(historyPanel).getAllByRole("listitem")).toHaveLength(2);
    expect(within(historyPanel).getByText("Edited human interpretation.")).toBeInTheDocument();
    expect(within(historyPanel).getByText("Note: Clarified scope.")).toBeInTheDocument();
  });

  it("uses Back locally without reload or relationship mutation", async () => {
    const onBack = vi.fn();
    const user = userEvent.setup();
    renderDetail({ onBack });

    await user.click(screen.getByRole("button", { name: /back/i }));

    expect(onBack).toHaveBeenCalledTimes(1);
    expect(proposePaperRelationship).not.toHaveBeenCalled();
  });
});
