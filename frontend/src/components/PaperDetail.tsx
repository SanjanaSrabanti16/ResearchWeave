import { useCallback, useEffect, useRef, useState } from "react";

import { getCachedPaperAnalysis } from "../api/client";
import type { CachedPaperAnalysisResponse, Paper } from "../types/paper";
import { PaperPdfPanel } from "./PaperPdfPanel";

interface Props {
  paper: Paper;
  onBack: () => void;
}

export function PaperDetail({ paper, onBack }: Props) {
  const [cached, setCached] = useState<CachedPaperAnalysisResponse | null>(null);
  const [cacheLoading, setCacheLoading] = useState(true);
  const requestGeneration = useRef(0);

  const refreshCachedState = useCallback(async (showLoading = false) => {
    const generation = ++requestGeneration.current;
    if (showLoading) setCacheLoading(true);
    try {
      const result = await getCachedPaperAnalysis(paper.id);
      if (requestGeneration.current === generation) setCached(result);
    } catch {
      // Cache lookup is optional; existing PDF and insight actions remain available.
    } finally {
      if (showLoading && requestGeneration.current === generation) setCacheLoading(false);
    }
  }, [paper.id]);

  useEffect(() => {
    setCached(null);
    void refreshCachedState(true);
    return () => {
      requestGeneration.current += 1;
    };
  }, [paper.id, refreshCachedState]);

  return (
    <aside className="paper-detail" aria-label={`Details for ${paper.title}`}>
      <header className="paper-detail-header">
        <button type="button" className="back-button" data-design-token="rw-periwinkle-lighter" onClick={onBack}>← Back</button>
      </header>
      <div className="paper-detail-scroll">
        <h2>{paper.title}</h2>
        <p className="metadata">
          {paper.authors.length ? paper.authors.join(", ") : "Authors unavailable"}
          {paper.publication_year ? ` · ${paper.publication_year}` : ""}
          {paper.venue ? ` · ${paper.venue}` : ""}
        </p>
        <h3>Abstract</h3>
        <p className="detail-abstract">{paper.abstract ?? "No abstract available."}</p>
        {cacheLoading && <p className="cache-status" role="status">Checking existing analysis…</p>}
        {!cacheLoading && (
          <PaperPdfPanel
            paper={paper}
            initialDocument={cached?.document}
            initialInsights={cached?.insights}
            initialProvider={cached?.insight_provider}
            onStateChanged={() => refreshCachedState(false)}
          />
        )}
      </div>
    </aside>
  );
}
