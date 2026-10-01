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
  pmcid?: string | null;
  pmcids?: string[];
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

export interface ProviderHealth {
  status: "ok" | "degraded" | "unavailable";
  successful_requests: number;
  failed_requests: number;
  cached_requests: number;
  message: string | null;
}

export interface SearchResponse {
  query: string;
  overall_status: "success" | "no_results";
  candidate_count: number;
  deduplicated_count: number;
  ranked_count: number;
  papers: Paper[];
  provider_status: Record<string, ProviderHealth>;
  warnings: string[];
}

export interface GraphNode {
  paper_id: string;
  title: string;
  query_relevance: number;
  node_weight: number;
  node_radius: number;
  information_completeness: number;
  node_opacity: number;
}

export type EdgeSelectionReason = "source_top_k" | "target_top_k" | "both_top_k";

export interface GraphEdge {
  source_paper_id: string;
  target_paper_id: string;
  paper_similarity: number;
  edge_weight: number;
  selection_reason: EdgeSelectionReason;
}

export interface RelatedConcept {
  text: string;
  query_similarity: number;
}

export interface GraphResponse {
  semantics_version: string;
  embedding_model: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  related_concepts: RelatedConcept[];
}

export interface GraphPayload {
  query: string;
  papers: Paper[];
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
    acquisition_method:
      | "existing_pdf"
      | "arxiv"
      | "unpaywall"
      | "crossref"
      | "pmc"
      | "publisher"
      | "upload";
    acquisition_provenance?:
      | "known_pdf_url"
      | "alternate_pdf_url"
      | "arxiv_id"
      | "unpaywall"
      | "crossref"
      | "pmc"
      | "publisher_landing_page"
      | "doi_landing_page"
      | "title_verified_arxiv_fallback"
      | "upload"
      | null;
    url: string | null;
    sha256: string;
    size_bytes: number;
  };
}

export interface PDFProcessingResponse {
  status:
    | "existing_pdf"
    | "arxiv"
    | "unpaywall"
    | "crossref"
    | "pmc"
    | "publisher"
    | "upload"
    | "upload_required";
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
  availability?:
    | "configured"
    | "unconfigured"
    | "available"
    | "temporarily_unavailable"
    | "authentication_error";
  message?: string;
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

export interface CachedPaperAnalysisResponse {
  paper_id: string;
  document: ParsedPaper | null;
  insights: InsightsResponse | null;
  insight_provider: LLMProviderId | null;
}

export type RelationshipType =
  | "shared_problem"
  | "shared_method"
  | "shared_finding"
  | "complementary_contribution"
  | "contrasting_result"
  | "shared_limitation"
  | "shared_future_work"
  | "related_application"
  | "other";

export interface RelationshipProposalPayload {
  source_paper_id: string;
  target_paper_id: string;
  requested_relationship_types?: RelationshipType[];
}

export interface RelationshipItem {
  relationship_type: RelationshipType;
  summary: string;
  source_evidence_ids: string[];
  target_evidence_ids: string[];
  confidence: number;
}

export interface RelationshipEvidence {
  paper_id: string;
  evidence_id: string;
  insight_field: string;
  claim: string;
  quote: string;
  section_id: string | null;
  section_heading: string | null;
  page_start: number | null;
  page_end: number | null;
}

export interface RelationshipProposal {
  proposal_id: string;
  source_paper_id: string;
  target_paper_id: string;
  semantic_similarity: number;
  relationship_types: RelationshipType[];
  relationships: RelationshipItem[];
  summary: string;
  evidence_source: RelationshipEvidence[];
  evidence_target: RelationshipEvidence[];
  confidence: number;
  source_analysis: {
    document_fingerprint: string;
    provider: string;
    model: string;
    extraction_version: string;
  };
  target_analysis: {
    document_fingerprint: string;
    provider: string;
    model: string;
    extraction_version: string;
  };
  relationship_provider: string;
  relationship_model: string;
  relationship_pipeline_version: string;
  created_at: string;
  cached: boolean;
  diagnostics: Array<{ item_index: number; reason: string }>;
}

export interface RelationshipReviewPayload {
  decision: "accepted" | "rejected" | "edited";
  edited_relationship_types?: RelationshipType[];
  edited_summary?: string;
  reviewer_note?: string;
}

export interface RelationshipReview {
  review_id: string;
  proposal_id: string;
  review_version: number;
  decision: "accepted" | "rejected" | "edited";
  original_relationship_types: RelationshipType[];
  original_summary: string;
  relationship_pipeline_version: string;
  edited_relationship_types: RelationshipType[] | null;
  edited_summary: string | null;
  reviewer_note: string | null;
  created_at: string;
}

export interface RelationshipReviewHistory {
  proposal_id: string;
  proposal: RelationshipProposal;
  reviews: RelationshipReview[];
}
