import { useMemo, useState } from "react";

import type { GraphEdge, GraphResponse, Paper } from "../types/paper";
import { orderPapersByQueryRelevance } from "../utils/graph";
import { GraphPaperList } from "./GraphPaperList";
import { PaperDetail } from "./PaperDetail";
import { RelationshipDetail } from "./RelationshipDetail";
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
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const selectedPaper = useMemo(
    () => papers.find((paper) => paper.id === selectedPaperId) ?? null,
    [papers, selectedPaperId],
  );
  const displayedPapers = useMemo(
    () => graph ? orderPapersByQueryRelevance(papers, graph.nodes) : papers,
    [graph, papers],
  );

  function hoverPaper(paperId: string | null) {
    setHoveredEdge(null);
    setHoveredPaperIds(paperId ? [paperId] : []);
  }

  function hoverEdge(edge: HoveredGraphEdge | null) {
    setHoveredEdge(edge);
    setHoveredPaperIds(edge ? [edge.sourcePaperId, edge.targetPaperId] : []);
  }

  function selectPaper(paperId: string) {
    setSelectedEdge(null);
    setSelectedPaperId(paperId);
  }

  function selectEdge(edge: GraphEdge) {
    setSelectedPaperId(null);
    setSelectedEdge(edge);
  }

  return (
    <section
      className="graph-workspace"
      data-desktop-layout="45-55"
      aria-label="Research paper graph workspace"
    >
      <div className="graph-panel" data-workspace-region="graph">
        {graph && (
          <ResearchGraph
            graph={graph}
            activePaperIds={hoveredPaperIds}
            hoveredEdge={hoveredEdge}
            selectedEdge={selectedEdge}
            selectedPaperId={selectedPaperId}
            onHoverPaper={hoverPaper}
            onHoverEdge={hoverEdge}
            onSelectEdge={selectEdge}
            onSelectPaper={selectPaper}
          />
        )}
        {graphLoading && <div className="graph-status" role="status">Building semantic graph…</div>}
        {!graph && !graphLoading && graphError && (
          <div className="graph-status graph-error" role="alert">
            Graph unavailable: {graphError}. Papers remain available.
          </div>
        )}
      </div>
      <div className="workspace-side-panel" data-workspace-region="papers">
        {selectedEdge ? (
          <RelationshipDetail
            edge={selectedEdge}
            sourcePaper={papers.find((paper) => paper.id === selectedEdge.source_paper_id)!}
            targetPaper={papers.find((paper) => paper.id === selectedEdge.target_paper_id)!}
            onBack={() => setSelectedEdge(null)}
            onOpenPaper={selectPaper}
          />
        ) : selectedPaper ? (
          <PaperDetail paper={selectedPaper} onBack={() => setSelectedPaperId(null)} />
        ) : (
          <GraphPaperList
            papers={displayedPapers}
            activePaperIds={hoveredPaperIds}
            edgeEndpointIds={
              hoveredEdge ? [hoveredEdge.sourcePaperId, hoveredEdge.targetPaperId] : []
            }
            selectedPaperId={selectedPaperId}
            onHoverPaper={hoverPaper}
            onSelectPaper={selectPaper}
          />
        )}
      </div>
    </section>
  );
}
