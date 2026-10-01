import type {
  CachedPaperAnalysisResponse,
  GraphPayload,
  GraphResponse,
  Paper,
  InsightsResponse,
  LLMProviderId,
  LLMProviderStatusResponse,
  ParsedPaper,
  PDFProcessingResponse,
  RelationshipProposal,
  RelationshipProposalPayload,
  RelationshipReview,
  RelationshipReviewHistory,
  RelationshipReviewPayload,
  SearchPayload,
  SearchResponse,
} from "../types/paper";

const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class APIError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = "APIError";
  }
}

export async function searchPapers(
  payload: SearchPayload,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const response = await fetch(`${API_URL}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    throw await apiError(response, "Search failed");
  }
  return (await response.json()) as SearchResponse;
}

export async function buildPaperGraph(
  payload: GraphPayload,
  signal?: AbortSignal,
): Promise<GraphResponse> {
  const response = await fetch(`${API_URL}/api/graph`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    throw await apiError(response, "Graph generation failed");
  }
  return (await response.json()) as GraphResponse;
}

export async function getCachedPaperAnalysis(
  paperId: string,
): Promise<CachedPaperAnalysisResponse> {
  const response = await fetch(
    `${API_URL}/api/papers/${encodeURIComponent(paperId)}/cached-analysis`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw await apiError(response, "Cached analysis lookup failed");
  }
  return (await response.json()) as CachedPaperAnalysisResponse;
}

async function apiError(response: Response, fallback: string): Promise<APIError> {
  let message = `${fallback} (${response.status})`;
  let code: string | undefined;
  try {
    const data = (await response.json()) as {
      detail?: string | Array<{ msg: string }> | { code?: string; message?: string };
    };
    if (typeof data.detail === "string") message = data.detail;
    else if (Array.isArray(data.detail)) message = data.detail.map((item) => item.msg).join("; ");
    else if (data.detail) {
      if (data.detail.message) message = data.detail.message;
      code = data.detail.code;
    }
  } catch {
    // Retain the status-based fallback when the server returns non-JSON content.
  }
  return new APIError(message, response.status, code);
}

export async function acquireAndParsePDF(paper: Paper): Promise<PDFProcessingResponse> {
  const response = await fetch(`${API_URL}/api/papers/pdf/acquire`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      paper_id: paper.id,
      doi: paper.doi,
      arxiv_id: paper.arxiv_id,
      arxiv_ids: paper.arxiv_ids ?? [],
      pmcid: paper.pmcid ?? null,
      pmcids: paper.pmcids ?? [],
      pdf_url: paper.pdf_url,
      alternate_pdf_urls: paper.alternate_pdf_urls ?? [],
      url: paper.url,
      alternate_urls: paper.alternate_urls ?? [],
      title: paper.title,
      authors: paper.authors,
      publication_year: paper.publication_year,
    }),
  });
  if (!response.ok) {
    throw await apiError(response, "PDF acquisition failed");
  }
  return (await response.json()) as PDFProcessingResponse;
}

export async function uploadAndParsePDF(
  paperId: string,
  file: File,
): Promise<PDFProcessingResponse> {
  const body = new FormData();
  body.append("paper_id", paperId);
  body.append("file", file);
  const response = await fetch(`${API_URL}/api/papers/pdf/upload`, {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw await apiError(response, "PDF upload failed");
  }
  return (await response.json()) as PDFProcessingResponse;
}

type InsightEvent =
  | { type: "stage"; stage: string }
  | { type: "result"; data: InsightsResponse }
  | { type: "error"; message: string };

export async function extractPaperInsights(
  document: ParsedPaper,
  onProgress?: (stage: string) => void,
  provider?: LLMProviderId,
): Promise<InsightsResponse> {
  const response = await fetch(`${API_URL}/api/papers/insights/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document, ...(provider ? { provider } : {}) }),
  });
  if (!response.ok) {
    throw await apiError(response, "Insight extraction failed");
  }
  if (!response.body) throw new Error("Insight progress stream is unavailable");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { done, value } = await reader.read();
    pending += decoder.decode(value, { stream: !done });
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line) as InsightEvent;
      if (event.type === "stage") onProgress?.(event.stage);
      if (event.type === "result") return event.data;
      if (event.type === "error") throw new Error(event.message);
    }
    if (done) break;
  }
  throw new Error("Insight progress stream ended before a result was returned");
}

export async function getLLMProviders(): Promise<LLMProviderStatusResponse> {
  const response = await fetch(`${API_URL}/api/llm/providers`);
  if (!response.ok) {
    throw await apiError(response, "Provider status failed");
  }
  return (await response.json()) as LLMProviderStatusResponse;
}

export async function proposePaperRelationship(
  payload: RelationshipProposalPayload,
): Promise<RelationshipProposal> {
  const response = await fetch(`${API_URL}/api/relationships/propose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw await apiError(response, "Relationship proposal failed");
  }
  return (await response.json()) as RelationshipProposal;
}

export async function getPaperRelationship(proposalId: string): Promise<RelationshipProposal> {
  const response = await fetch(`${API_URL}/api/relationships/${encodeURIComponent(proposalId)}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw await apiError(response, "Relationship lookup failed");
  }
  return (await response.json()) as RelationshipProposal;
}

export async function reviewPaperRelationship(
  proposalId: string,
  payload: RelationshipReviewPayload,
): Promise<RelationshipReview> {
  const response = await fetch(
    `${API_URL}/api/relationships/${encodeURIComponent(proposalId)}/review`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
  if (!response.ok) {
    throw await apiError(response, "Relationship review failed");
  }
  return (await response.json()) as RelationshipReview;
}

export async function getPaperRelationshipHistory(
  proposalId: string,
): Promise<RelationshipReviewHistory> {
  const response = await fetch(
    `${API_URL}/api/relationships/${encodeURIComponent(proposalId)}/history`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw await apiError(response, "Relationship history failed");
  }
  return (await response.json()) as RelationshipReviewHistory;
}

export async function getPaperRelationshipByPair(
  sourcePaperId: string,
  targetPaperId: string,
): Promise<RelationshipProposal | null> {
  const params = new URLSearchParams({
    source_paper_id: sourcePaperId,
    target_paper_id: targetPaperId,
  });
  const response = await fetch(`${API_URL}/api/relationships/by-pair?${params}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw await apiError(response, "Relationship cache lookup failed");
  }
  return (await response.json()) as RelationshipProposal | null;
}
