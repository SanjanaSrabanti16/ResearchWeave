import type { InsightFields } from "../types/paper";

const sections: Array<[keyof InsightFields, string]> = [
  ["paper_overview", "Paper Overview"],
  ["research_problem", "Problem"],
  ["methods", "Methods"],
  ["key_contributions", "Contributions"],
  ["evaluation", "Evaluation"],
  ["main_findings", "Findings"],
  ["why_it_matters", "Why It Matters"],
  ["target_audience", "Audience"],
  ["limitations", "Limitations"],
  ["future_work", "Future Work"],
];

export function PaperInsights({ insights }: { insights: InsightFields }) {
  return (
    <section className="paper-insights" aria-label="Grounded research insights">
      <h4>Research insights</h4>
      {sections.map(([field, label]) => (
        <div className="insight-field" key={field}>
          <h5>{label}</h5>
          {insights[field].length ? (
            <ul>
              {insights[field].map((item, index) => (
                <li key={`${field}-${index}`}>
                  <p className={field === "paper_overview" ? "paper-overview-text" : undefined}>
                    {item.claim}
                  </p>
                  <details>
                    <summary>Evidence ({item.evidence.length})</summary>
                    <ul>
                      {item.evidence.map((reference, evidenceIndex) => (
                        <li key={`${reference.chunk_id}-${evidenceIndex}`}>
                          <blockquote>“{reference.quote}”</blockquote>
                          <code>{reference.chunk_id}</code>
                        </li>
                      ))}
                    </ul>
                  </details>
                </li>
              ))}
            </ul>
          ) : (
            <p>
              {field === "limitations"
                ? "The paper does not explicitly state study limitations."
                : "No grounded insight extracted."}
            </p>
          )}
        </div>
      ))}
    </section>
  );
}
