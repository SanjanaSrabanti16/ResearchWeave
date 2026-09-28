import { FormEvent, useEffect, useRef, useState } from "react";

import { buildPaperGraph, searchPapers } from "../api/client";
import { GraphWorkspace } from "../components/GraphWorkspace";
import type { GraphResponse, SearchResponse } from "../types/paper";

const currentYear = new Date().getFullYear();

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [startYear, setStartYear] = useState("");
  const [endYear, setEndYear] = useState("");
  const [limit, setLimit] = useState(20);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setGraphError(null);
    if (startYear && endYear && Number(startYear) > Number(endYear)) {
      setError("Start year must be before or equal to end year.");
      return;
    }
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setGraph(null);
    void searchPapers(
      {
        query,
        limit,
        ...(startYear ? { start_year: Number(startYear) } : {}),
        ...(endYear ? { end_year: Number(endYear) } : {}),
      },
      controller.signal,
    )
      .then((searchResult) => {
        setResult(searchResult);
        setLoading(false);
        if (!searchResult.papers.length) return;
        setGraphLoading(true);
        return buildPaperGraph(
          { query: searchResult.query, papers: searchResult.papers },
          controller.signal,
        )
          .then(setGraph)
          .catch((caught: unknown) => {
            if (
              typeof caught === "object" &&
              caught !== null &&
              "name" in caught &&
              caught.name === "AbortError"
            ) return;
            setGraphError(
              caught instanceof Error ? caught.message : "Graph generation failed unexpectedly.",
            );
          })
          .finally(() => {
            if (controllerRef.current === controller) setGraphLoading(false);
          });
      })
      .catch((caught: unknown) => {
        if (
          typeof caught === "object" &&
          caught !== null &&
          "name" in caught &&
          caught.name === "AbortError"
        ) return;
        setError(caught instanceof Error ? caught.message : "Search failed unexpectedly.");
      })
      .finally(() => {
        if (controllerRef.current === controller) setLoading(false);
      });
  }

  return (
    <main className="application-shell">
      <header className="app-header">
        <div className="brand-block">
          <h1>ResearchWeave</h1>
          <p>Explore a query-centered scholarly landscape.</p>
        </div>
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
          <label>
            Start year
            <input type="number" min="1800" max="2100" value={startYear} onChange={(event) => setStartYear(event.target.value)} placeholder="Any" />
          </label>
          <label>
            End year
            <input type="number" min="1800" max="2100" value={endYear} onChange={(event) => setEndYear(event.target.value)} placeholder={String(currentYear)} />
          </label>
          <label>
            Papers
            <input type="number" min="5" max="50" value={limit} onChange={(event) => setLimit(Number(event.target.value))} required />
          </label>
          <button type="submit" className="search-button" data-design-token="rw-periwinkle" disabled={loading}>{loading ? "Searching…" : "Search"}</button>
        </form>
      </header>

      {loading && <div className="status loading" role="status">Searching papers…</div>}
      {error && <div className="status error" role="alert">{error}</div>}

      {result && !loading && (
        <section className="search-output" aria-live="polite">
          <div className="result-status-row">
            <span>{result.papers.length} ranked papers</span>
            {Object.entries(result.provider_status).map(([provider, health]) => (
              <span key={provider} className={`provider ${health.status}`}>
                {provider.replace("_", " ")}: {health.status}
                {` (${health.successful_requests} succeeded, ${health.failed_requests} failed)`}
              </span>
            ))}
          </div>
          {Object.entries(result.provider_status).map(([provider, health]) =>
            health.message ? (
              <div className="status warning" key={`${provider}-health`}>
                {provider.replace("_", " ")}: {health.message}
              </div>
            ) : null,
          )}
          {result.warnings.map((warning) => <div className="status warning" key={warning}>{warning}</div>)}
          {result.papers.length ? (
            <GraphWorkspace
              papers={result.papers}
              graph={graph}
              graphLoading={graphLoading}
              graphError={graphError}
            />
          ) : (
            <div className="status">No matching papers were found.</div>
          )}
        </section>
      )}
    </main>
  );
}
