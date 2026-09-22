import type { Paper } from "../types/paper";
import { formatScore } from "../utils/format";
import { PaperPdfPanel } from "./PaperPdfPanel";

interface Props {
  paper: Paper;
  rank: number;
}

export function PaperCard({ paper, rank }: Props) {
  const primaryUrl = paper.url ?? (paper.arxiv_id ? `https://arxiv.org/abs/${paper.arxiv_id}` : null);

  return (
    <article className="paper-card">
      <div className="rank" aria-label={`Rank ${rank}`}>
        {rank}
      </div>
      <div className="paper-content">
        <h2>
          {primaryUrl ? (
            <a href={primaryUrl} target="_blank" rel="noreferrer">
              {paper.title}
            </a>
          ) : (
            paper.title
          )}
        </h2>
        <p className="metadata">
          {paper.authors.length ? paper.authors.join(", ") : "Authors unavailable"}
          {paper.publication_year ? ` · ${paper.publication_year}` : ""}
          {paper.venue ? ` · ${paper.venue}` : ""}
        </p>
        <p className="abstract">{paper.abstract ?? "No abstract available."}</p>
        <div className="details">
          <span>Citations: {paper.citation_count ?? "—"}</span>
          <span>Semantic: {formatScore(paper.semantic_score)}</span>
          <span>Reranker: {formatScore(paper.reranker_score)}</span>
        </div>
        <div className="links">
          {paper.source_names.map((source) => (
            <span className="source" key={source}>{source.replace("_", " ")}</span>
          ))}
          {paper.doi && (
            <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer">DOI</a>
          )}
          {paper.pdf_url && (
            <a href={paper.pdf_url} target="_blank" rel="noreferrer">Open PDF</a>
          )}
        </div>
        <PaperPdfPanel paper={paper} />
      </div>
    </article>
  );
}
