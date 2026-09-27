import type {
  Paper,
  InsightsResponse,
  LLMProviderId,
  LLMProviderStatusResponse,
  ParsedPaper,
  PDFProcessingResponse,
  SearchPayload,
  SearchResponse,
} from "../types/paper";

const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class APIError extends Error {
  constructor(message: string, readonly status: number) {
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
    throw new APIError(await responseError(response, "Search failed"), response.status);
  }
  return (await response.json()) as SearchResponse;
}

async function responseError(response: Response, fallback: string): Promise<string> {
  let message = `${fallback} (${response.status})`;
  try {
    const data = (await response.json()) as { detail?: string | Array<{ msg: string }> };
    if (typeof data.detail === "string") message = data.detail;
    else if (Array.isArray(data.detail)) message = data.detail.map((item) => item.msg).join("; ");
  } catch {
    // Retain the status-based fallback when the server returns non-JSON content.
  }
  return message;
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
    throw new APIError(await responseError(response, "PDF acquisition failed"), response.status);
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
    throw new APIError(await responseError(response, "PDF upload failed"), response.status);
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
    throw new APIError(await responseError(response, "Insight extraction failed"), response.status);
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
    throw new APIError(await responseError(response, "Provider status failed"), response.status);
  }
  return (await response.json()) as LLMProviderStatusResponse;
}
