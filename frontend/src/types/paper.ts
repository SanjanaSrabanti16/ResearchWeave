export interface Paper {
  id: string;
  title: string;
  abstract: string | null;
  authors: string[];
  publication_year: number | null;
  publication_date: string | null;
  venue: string | null;
  doi: string | null;
  arxiv_id: string | null;
  arxiv_ids?: string[];
  openalex_id: string | null;
  semantic_scholar_id: string | null;
  url: string | null;
  alternate_urls?: string[];
  pdf_url: string | null;
  alternate_pdf_urls?: string[];
  citation_count: number | null;
  source_names: string[];
  semantic_score: number | null;
  reranker_score: number | null;
}

export interface SearchPayload {
  query: string;
  start_year?: number;
  end_year?: number;
  limit: number;
}

export interface SearchResponse {
  query: string;
  candidate_count: number;
  deduplicated_count: number;
  ranked_count: number;
  papers: Paper[];
  provider_status: Record<string, string>;
  warnings: string[];
}

export interface DocumentChunk {
  id: string;
  section_id: string;
  text: string;
  start_char: number;
  end_char: number;
  page_start: number | null;
  page_end: number | null;
}

export interface DocumentSection {
  id: string;
  heading: string | null;
  level: number;
  text: string;
  page_start: number | null;
  page_end: number | null;
  chunks: DocumentChunk[];
}

export interface ParsedPaper {
  paper_id: string;
  title: string | null;
  authors: string[];
  abstract: string | null;
  abstract_chunks: DocumentChunk[];
  sections: DocumentSection[];
  references: Array<Record<string, unknown>>;
  parser: string;
  parser_version: string;
  source_pdf: {
    acquisition_method: "existing_pdf" | "arxiv" | "unpaywall" | "upload";
    acquisition_provenance?:
      | "known_pdf_url"
      | "alternate_pdf_url"
      | "arxiv_id"
      | "unpaywall"
      | "title_verified_arxiv_fallback"
      | "upload"
      | null;
    url: string | null;
    sha256: string;
    size_bytes: number;
  };
}

export interface PDFProcessingResponse {
  status: "existing_pdf" | "arxiv" | "unpaywall" | "upload" | "upload_required";
  paper_id: string;
  document: ParsedPaper | null;
  message: string | null;
  acquisition_provenance?: string | null;
  attempts?: Array<{
    route: string;
    candidate_type: string;
    safe_identifier: string;
    outcome: string;
  }>;
}

export interface EvidenceReference {
  chunk_id: string;
  quote: string;
}

export interface InsightClaim {
  claim: string;
  evidence: EvidenceReference[];
}

export interface InsightFields {
  paper_overview: InsightClaim[];
  research_problem: InsightClaim[];
  methods: InsightClaim[];
  key_contributions: InsightClaim[];
  evaluation: InsightClaim[];
  main_findings: InsightClaim[];
  why_it_matters: InsightClaim[];
  target_audience: InsightClaim[];
  limitations: InsightClaim[];
  future_work: InsightClaim[];
}

export type LLMProviderId = "ollama" | "gemini" | "evl_gemma";

export interface LLMProviderStatus {
  provider_id: LLMProviderId;
  model: string;
  configured: boolean;
  cloud: boolean;
}

export interface LLMProviderStatusResponse {
  default_provider: LLMProviderId;
  providers: LLMProviderStatus[];
}

export interface InsightsResponse {
  paper_id: string;
  document_fingerprint: string;
  model: string;
  extraction_version: string;
  cached: boolean;
  insights: InsightFields;
}
