import { FormEvent, useEffect, useRef, useState } from "react";

import { searchPapers } from "../api/client";
import { PaperCard } from "../components/PaperCard";
import type { SearchResponse } from "../types/paper";

const currentYear = new Date().getFullYear();

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [startYear, setStartYear] = useState("");
  const [endYear, setEndYear] = useState("");
  const [limit, setLimit] = useState(20);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    if (startYear && endYear && Number(startYear) > Number(endYear)) {
      setError("Start year must be before or equal to end year.");
      return;
    }
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    void searchPapers(
      {
        query,
        limit,
        ...(startYear ? { start_year: Number(startYear) } : {}),
        ...(endYear ? { end_year: Number(endYear) } : {}),
      },
      controller.signal,
    )
      .then(setResult)
      .catch((caught: unknown) => {
        if (
          typeof caught === "object" &&
          caught !== null &&
          "name" in caught &&
          caught.name === "AbortError"
        ) {
          return;
        }
        setError(caught instanceof Error ? caught.message : "Search failed unexpectedly.");
      })
      .finally(() => {
        if (controllerRef.current === controller) setLoading(false);
      });
  }

  return (
    <main>
      <section className="hero">
        <p className="eyebrow">Open scholarly discovery</p>
        <h1>Research Landscape Explorer</h1>
        <p>Search three open scholarly sources and rank the literature by relevance to your question.</p>
      </section>

      <form className="search-form" onSubmit={submit}>
        <label className="topic-field">
          Research topic
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. AI agents for visualization systems"
            minLength={2}
            maxLength={500}
            required
          />
        </label>
        <div className="filters">
          <label>
            Start year
            <input type="number" min="1800" max="2100" value={startYear} onChange={(e) => setStartYear(e.target.value)} placeholder="Any" />
          </label>
          <label>
            End year
            <input type="number" min="1800" max="2100" value={endYear} onChange={(e) => setEndYear(e.target.value)} placeholder={String(currentYear)} />
          </label>
          <label>
            Results
            <input type="number" min="5" max="50" value={limit} onChange={(e) => setLimit(Number(e.target.value))} required />
          </label>
          <button type="submit" disabled={loading}>{loading ? "Ranking…" : "Search papers"}</button>
        </div>
      </form>

      {loading && <div className="status loading" role="status">Retrieving and ranking papers. The first search may take several minutes while models download.</div>}
      {error && <div className="status error" role="alert">{error}</div>}

      {result && !loading && (
        <section className="results" aria-live="polite">
          <div className="summary">
            <div><strong>{result.candidate_count}</strong><span>candidates</span></div>
            <div><strong>{result.deduplicated_count}</strong><span>unique papers</span></div>
            <div><strong>{result.ranked_count}</strong><span>ranked results</span></div>
          </div>
          <div className="provider-status">
            {Object.entries(result.provider_status).map(([provider, status]) => (
              <span key={provider} className={`provider ${status === "error" ? "failed" : ""}`}>
                {provider.replace("_", " ")}: {status}
              </span>
            ))}
          </div>
          {result.warnings.map((warning) => <div className="status warning" key={warning}>{warning}</div>)}
          <div className="score-help">
            <p><strong>Semantic Score:</strong> Bi-encoder similarity between the research query and the paper.</p>
            <p><strong>Reranker Score:</strong> Raw model relevance score used to order papers. Higher values indicate greater relevance within this ranking model; scores are not directly comparable across different models.</p>
          </div>
          {result.papers.length ? result.papers.map((paper, index) => (
            <PaperCard key={paper.id} paper={paper} rank={index + 1} />
          )) : <div className="status">No matching papers were found.</div>}
        </section>
      )}
    </main>
  );
}
