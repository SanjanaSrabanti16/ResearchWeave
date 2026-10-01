import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { InsightFields } from "../types/paper";
import { PaperInsights } from "./PaperInsights";

function emptyInsights(): InsightFields {
  return {
    paper_overview: [],
    research_problem: [],
    methods: [],
    key_contributions: [],
    evaluation: [],
    main_findings: [],
    why_it_matters: [],
    target_audience: [],
    limitations: [],
    future_work: [],
  };
}

describe("PaperInsights", () => {
  it("keeps rich uncited prose while hiding an empty evidence disclosure", () => {
    const insights = emptyInsights();
    insights.paper_overview = [{
      claim: "A complete researcher explanation remains visible even when no reliable citation was returned.",
      evidence: [],
    }];

    const { container } = render(<PaperInsights insights={insights} />);

    expect(screen.getByText(insights.paper_overview[0].claim)).toBeInTheDocument();
    expect(container.querySelector(".insight-evidence")).toBeNull();
    expect(screen.getByText("The paper does not explicitly specify a target audience.")).toBeInTheDocument();
    expect(screen.getByText("No explicit future-work directions were identified.")).toBeInTheDocument();
  });

  it("shows evidence only when a valid reference is present", () => {
    const insights = emptyInsights();
    insights.methods = [{
      claim: "The system coordinates multiple analytical views.",
      evidence: [{ chunk_id: "method-1", quote: "The system coordinates multiple views." }],
    }];

    render(<PaperInsights insights={insights} />);

    expect(screen.getByText("Evidence (1)")).toBeInTheDocument();
    expect(screen.getByText("method-1")).toBeInTheDocument();
  });
});
