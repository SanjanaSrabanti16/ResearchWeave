import { useMemo, useState } from "react";

import type { GraphResponse, Paper } from "../types/paper";
import { GraphPaperList } from "./GraphPaperList";
import { PaperDetail } from "./PaperDetail";
import { type HoveredGraphEdge, ResearchGraph } from "./ResearchGraph";

interface Props {
  papers: Paper[];
  graph: GraphResponse | null;
  graphLoading: boolean;
  graphError: string | null;
}

export function GraphWorkspace({ papers, graph, graphLoading, graphError }: Props) {
  const [hoveredPaperIds, setHoveredPaperIds] = useState<string[]>([]);
  const [hoveredEdge, setHoveredEdge] = useState<HoveredGraphEdge | null>(null);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);
  const selectedPaper = useMemo(
    () => papers.find((paper) => paper.id === selectedPaperId) ?? null,
    [papers, selectedPaperId],
  );

  function hoverPaper(paperId: string | null) {
    setHoveredEdge(null);
    setHoveredPaperIds(paperId ? [paperId] : []);
  }

  function hoverEdge(edge: HoveredGraphEdge | null) {
    setHoveredEdge(edge);
    setHoveredPaperIds(edge ? [edge.sourcePaperId, edge.targetPaperId] : []);
  }

  return (
    <section className="graph-workspace" aria-label="Research paper graph workspace">
      <div className="graph-panel">
        {graph && (
          <ResearchGraph
            graph={graph}
            activePaperIds={hoveredPaperIds}
            hoveredEdge={hoveredEdge}
            selectedPaperId={selectedPaperId}
            onHoverPaper={hoverPaper}
            onHoverEdge={hoverEdge}
            onSelectPaper={setSelectedPaperId}
          />
        )}
        {graphLoading && <div className="graph-status" role="status">Building semantic graph…</div>}
        {!graph && !graphLoading && graphError && (
          <div className="graph-status graph-error" role="alert">
            Graph unavailable: {graphError}. Papers remain available.
          </div>
        )}
      </div>
      <div className="workspace-side-panel">
        {selectedPaper ? (
          <PaperDetail paper={selectedPaper} onBack={() => setSelectedPaperId(null)} />
        ) : (
          <GraphPaperList
            papers={papers}
            activePaperIds={hoveredPaperIds}
            edgeEndpointIds={
              hoveredEdge ? [hoveredEdge.sourcePaperId, hoveredEdge.targetPaperId] : []
            }
            selectedPaperId={selectedPaperId}
            onHoverPaper={hoverPaper}
            onSelectPaper={setSelectedPaperId}
          />
        )}
      </div>
    </section>
  );
}
