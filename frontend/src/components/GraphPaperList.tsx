import { useEffect, useRef } from "react";

import type { Paper } from "../types/paper";

interface Props {
  papers: Paper[];
  activePaperIds: string[];
  edgeEndpointIds: string[];
  selectedPaperId: string | null;
  onHoverPaper: (paperId: string | null) => void;
  onSelectPaper: (paperId: string) => void;
}

export function GraphPaperList({
  papers,
  activePaperIds,
  edgeEndpointIds,
  selectedPaperId,
  onHoverPaper,
  onSelectPaper,
}: Props) {
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (activePaperIds.length !== 1) return;
    const row = Array.from(
      listRef.current?.querySelectorAll<HTMLElement>("[data-paper-row-id]") ?? [],
    ).find((item) => item.dataset.paperRowId === activePaperIds[0]);
    row?.scrollIntoView?.({ block: "nearest" });
  }, [activePaperIds]);

  const active = new Set(activePaperIds);
  const edgeEndpoints = new Set(edgeEndpointIds);
  return (
    <section className="paper-list-panel" aria-label="Search result papers">
      <header><h2>Papers ({papers.length})</h2></header>
      <div className="paper-list-scroll" ref={listRef}>
        {papers.map((paper, index) => (
          <button
            type="button"
            className={`paper-list-row ${active.has(paper.id) ? "is-active" : ""} ${edgeEndpoints.has(paper.id) ? "is-edge-endpoint" : ""} ${selectedPaperId === paper.id ? "is-selected" : ""}`}
            data-paper-row-id={paper.id}
            data-highlight-token="rw-periwinkle-light"
            key={paper.id}
            onMouseEnter={() => onHoverPaper(paper.id)}
            onMouseLeave={() => onHoverPaper(null)}
            onFocus={() => onHoverPaper(paper.id)}
            onBlur={() => onHoverPaper(null)}
            onClick={() => onSelectPaper(paper.id)}
          >
            <span className="paper-row-rank">{index + 1}</span>
            <span className="paper-row-content">
              <strong>{paper.title}</strong>
              <span className="paper-row-metadata">
                {paper.authors.length ? paper.authors.slice(0, 3).join(", ") : "Authors unavailable"}
                {paper.authors.length > 3 ? " et al." : ""}
                {paper.publication_year ? ` · ${paper.publication_year}` : ""}
              </span>
              <span className="paper-row-abstract">
                {paper.abstract ?? "No abstract available."}
              </span>
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
