import type { InsightFields } from "../types/paper";

const sections: Array<[keyof InsightFields, string]> = [
  ["paper_overview", "Paper Overview"],
  ["research_problem", "Research Problem"],
  ["methods", "Methods"],
  ["key_contributions", "Key Contributions"],
  ["evaluation", "Evaluation"],
  ["main_findings", "Main Findings"],
  ["why_it_matters", "Why It Matters"],
  ["target_audience", "Target Audience"],
  ["limitations", "Limitations / Open Challenges"],
  ["future_work", "Future Work"],
];

function InsightIcon({ field }: { field: keyof InsightFields }) {
  const common = {
    className: "insight-card-icon",
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  if (field === "paper_overview") {
    return <svg {...common}><path d="M6 3.5h8l4 4V20.5H6z" /><path d="M14 3.5v4h4M9 12h6M9 15.5h6" /></svg>;
  }
  if (field === "research_problem") {
    return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M9.8 9a2.4 2.4 0 1 1 3.5 2.15c-.85.45-1.3.95-1.3 1.85M12 17h.01" /></svg>;
  }
  if (field === "methods") {
    return <svg {...common}><circle cx="12" cy="12" r="3" /><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1" /></svg>;
  }
  if (field === "key_contributions") {
    return <svg {...common}><path d="m12 3 2.7 5.4 6 .9-4.35 4.2 1.03 5.95L12 16.65l-5.38 2.8 1.03-5.95L3.3 9.3l6-.9z" /></svg>;
  }
  if (field === "evaluation") {
    return <svg {...common}><path d="M8 4h10v16H6V6z" /><path d="m8 11 1.5 1.5L12 10M14 11h2M8 16h8" /></svg>;
  }
  if (field === "main_findings") {
    return <svg {...common}><path d="M4 20V10h4v10M10 20V4h4v16M16 20v-7h4v7M3 20h18" /></svg>;
  }
  if (field === "why_it_matters") {
    return <svg {...common}><path d="M9 18h6M9.5 21h5M8.2 14.5A6 6 0 1 1 15.8 14.5c-.9.7-1.3 1.5-1.3 2.5h-5c0-1-.4-1.8-1.3-2.5Z" /></svg>;
  }
  if (field === "target_audience") {
    return <svg {...common}><circle cx="9" cy="8" r="3" /><circle cx="17" cy="9" r="2.3" /><path d="M3.5 20c.4-4 2.2-6 5.5-6s5.1 2 5.5 6M14 15c3.6-.5 5.7 1.2 6.5 4" /></svg>;
  }
  if (field === "limitations") {
    return <svg {...common}><path d="M12 3 2.8 20h18.4z" /><path d="M12 9v5M12 17h.01" /></svg>;
  }
  return <svg {...common}><path d="M5 19 19 5M11 5h8v8" /><path d="M5 7v12h12" /></svg>;
}

export function PaperInsights({ insights }: { insights: InsightFields }) {
  return (
    <section className="paper-insights" aria-label="Grounded research insights">
      <h4>Research insights</h4>
      {sections.map(([field, label]) => (
        <article className="insight-card" data-insight-field={field} data-surface-token="rw-periwinkle-light" key={field}>
          <header className="insight-card-header">
            <InsightIcon field={field} />
            <h5>{label}</h5>
          </header>
          {insights[field].length ? (
            <ul className={`insight-claim-list ${field === "paper_overview" ? "is-overview" : ""}`}>
              {insights[field].map((item, index) => (
                <li key={`${field}-${index}`}>
                  <p className={field === "paper_overview" ? "paper-overview-text" : "insight-claim-text"}>
                    {item.claim}
                  </p>
                  {item.evidence.length > 0 && <details className="insight-evidence">
                    <summary>Evidence ({item.evidence.length})</summary>
                    <ul>
                      {item.evidence.map((reference, evidenceIndex) => (
                        <li key={`${reference.chunk_id}-${evidenceIndex}`}>
                          <blockquote>“{reference.quote}”</blockquote>
                          <code>{reference.chunk_id}</code>
                        </li>
                      ))}
                    </ul>
                  </details>}
                </li>
              ))}
            </ul>
          ) : (
            <p className="insight-empty-state">
              {field === "limitations"
                ? "The paper does not explicitly state study limitations."
                : field === "target_audience"
                  ? "The paper does not explicitly specify a target audience."
                  : field === "future_work"
                    ? "No explicit future-work directions were identified."
                    : "No grounded insight extracted."}
            </p>
          )}
        </article>
      ))}
    </section>
  );
}
