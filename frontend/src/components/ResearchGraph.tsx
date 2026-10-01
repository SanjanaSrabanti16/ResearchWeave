import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";

import type { GraphEdge, GraphNode, GraphResponse } from "../types/paper";
import { edgeWidth, MIN_EDGE_SIMILARITY } from "../utils/graph";

interface SimulationNode extends GraphNode, d3.SimulationNodeDatum {}
interface SimulationEdge extends d3.SimulationLinkDatum<SimulationNode> {
  edge: GraphEdge;
}

interface TooltipState {
  x: number;
  y: number;
  content: React.ReactNode;
}

export interface HoveredGraphEdge {
  sourcePaperId: string;
  targetPaperId: string;
}

interface Props {
  graph: GraphResponse;
  activePaperIds: string[];
  hoveredEdge: HoveredGraphEdge | null;
  selectedEdge: GraphEdge | null;
  selectedPaperId: string | null;
  onHoverPaper: (paperId: string | null) => void;
  onHoverEdge: (edge: HoveredGraphEdge | null) => void;
  onSelectEdge: (edge: GraphEdge) => void;
  onSelectPaper: (paperId: string) => void;
}

function informationLabel(value: number): string {
  if (value >= 1) return "Insights available";
  if (value >= 0.78) return "PDF parsed";
  if (value >= 0.55) return "Abstract available";
  return "Metadata only";
}

function endpointId(endpoint: string | number | SimulationNode): string {
  return typeof endpoint === "object" ? endpoint.paper_id : String(endpoint);
}

export function ResearchGraph({
  graph,
  activePaperIds,
  hoveredEdge,
  selectedEdge,
  selectedPaperId,
  onHoverPaper,
  onHoverEdge,
  onSelectEdge,
  onSelectPaper,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const handlersRef = useRef({ onHoverPaper, onHoverEdge, onSelectEdge, onSelectPaper });
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  useEffect(() => {
    handlersRef.current = { onHoverPaper, onHoverEdge, onSelectEdge, onSelectPaper };
  }, [onHoverEdge, onHoverPaper, onSelectEdge, onSelectPaper]);

  useEffect(() => {
    const element = svgRef.current;
    if (!element) return;
    const width = element.clientWidth || 760;
    const height = element.clientHeight || 620;
    const svg = d3.select(element);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const viewport = svg.append("g").attr("class", "graph-viewport");
    const nodes: SimulationNode[] = graph.nodes.map((node) => ({ ...node }));
    const links: SimulationEdge[] = graph.edges.map((edge) => ({
      source: edge.source_paper_id,
      target: edge.target_paper_id,
      edge,
    }));
    const nodeById = new Map(nodes.map((node) => [node.paper_id, node]));

    const linkSelection = viewport
      .append("g")
      .attr("class", "graph-edges")
      .selectAll<SVGLineElement, SimulationEdge>("line")
      .data(links)
      .join("line")
      .attr("class", "graph-edge")
      .attr("role", "button")
      .attr("tabindex", 0)
      .attr("aria-label", (link) => {
        const source = nodeById.get(link.edge.source_paper_id)?.title ?? "Paper A";
        const target = nodeById.get(link.edge.target_paper_id)?.title ?? "Paper B";
        return `Open relationship between ${source} and ${target}`;
      })
      .attr("data-color-token", "rw-edge")
      .attr("data-source-id", (link) => link.edge.source_paper_id)
      .attr("data-target-id", (link) => link.edge.target_paper_id)
      .attr("data-edge-weight", (link) => link.edge.edge_weight)
      .style("cursor", "pointer")
      .attr("stroke-opacity", 1)
      .attr("stroke-width", (link) => edgeWidth(link.edge.edge_weight))
      .on("mouseenter", (event, link) => {
        handlersRef.current.onHoverEdge({
          sourcePaperId: link.edge.source_paper_id,
          targetPaperId: link.edge.target_paper_id,
        });
        const source = nodeById.get(link.edge.source_paper_id);
        const target = nodeById.get(link.edge.target_paper_id);
        setTooltip({
          x: event.offsetX + 12,
          y: event.offsetY + 12,
          content: (
            <>
              <strong>{source?.title}</strong>
              <span aria-hidden="true"> ↔ </span>
              <strong>{target?.title}</strong>
              <span>Semantic similarity: {link.edge.paper_similarity.toFixed(2)}</span>
            </>
          ),
        });
      })
      .on("mouseleave", () => {
        handlersRef.current.onHoverEdge(null);
        setTooltip(null);
      })
      .on("click", (event, link) => {
        if (!event.defaultPrevented) handlersRef.current.onSelectEdge(link.edge);
      })
      .on("keydown", (event, link) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handlersRef.current.onSelectEdge(link.edge);
        }
      });

    const nodeSelection = viewport
      .append("g")
      .attr("class", "graph-nodes")
      .selectAll<SVGCircleElement, SimulationNode>("circle")
      .data(nodes, (node) => node.paper_id)
      .join("circle")
      .attr("class", "graph-node")
      .attr("data-color-token", "rw-periwinkle")
      .attr("role", "button")
      .attr("tabindex", 0)
      .attr("aria-label", (node) => `Open ${node.title}`)
      .attr("data-paper-id", (node) => node.paper_id)
      .attr("data-node-radius", (node) => node.node_radius)
      .attr("data-node-opacity", (node) => node.node_opacity)
      .attr("r", (node) => node.node_radius)
      .attr("opacity", (node) => node.node_opacity)
      .on("mouseenter", (event, node) => {
        handlersRef.current.onHoverPaper(node.paper_id);
        setTooltip({
          x: event.offsetX + 12,
          y: event.offsetY + 12,
          content: (
            <>
              <strong>{node.title}</strong>
              <span>Query relevance: {node.query_relevance.toFixed(2)}</span>
              <span>{informationLabel(node.information_completeness)}</span>
            </>
          ),
        });
      })
      .on("mouseleave", () => {
        handlersRef.current.onHoverPaper(null);
        setTooltip(null);
      })
      .on("click", (event, node) => {
        if (!event.defaultPrevented) handlersRef.current.onSelectPaper(node.paper_id);
      })
      .on("keydown", (event, node) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handlersRef.current.onSelectPaper(node.paper_id);
        }
      });

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink<SimulationNode, SimulationEdge>(links)
          .id((node) => node.paper_id)
          .distance(105)
          .strength(0.35),
      )
      .force("charge", d3.forceManyBody().strength(-230))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide<SimulationNode>().radius((node) => node.node_radius + 6))
      .alphaDecay(0.055)
      .on("tick", () => {
        linkSelection
          .attr("x1", (link) => (link.source as SimulationNode).x ?? 0)
          .attr("y1", (link) => (link.source as SimulationNode).y ?? 0)
          .attr("x2", (link) => (link.target as SimulationNode).x ?? 0)
          .attr("y2", (link) => (link.target as SimulationNode).y ?? 0);
        nodeSelection
          .attr("cx", (node) => node.x ?? 0)
          .attr("cy", (node) => node.y ?? 0);
      });

    nodeSelection.call(
      d3
        .drag<SVGCircleElement, SimulationNode>()
        .on("start", (event, node) => {
          if (!event.active) simulation.alphaTarget(0.22).restart();
          node.fx = node.x;
          node.fy = node.y;
        })
        .on("drag", (event, node) => {
          node.fx = event.x;
          node.fy = event.y;
        })
        .on("end", (event, node) => {
          if (!event.active) simulation.alphaTarget(0);
          node.fx = null;
          node.fy = null;
        }),
    );

    svg.call(
      d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.45, 3])
        .on("zoom", (event) => viewport.attr("transform", event.transform)),
    );

    return () => {
      simulation.stop();
    };
  }, [graph]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const active = new Set(activePaperIds);
    const hasActive = active.size > 0;
    const hasHoveredEdge = hoveredEdge !== null;
    const selectedIds = new Set(
      selectedEdge ? [selectedEdge.source_paper_id, selectedEdge.target_paper_id] : [],
    );
    svg
      .selectAll<SVGCircleElement, SimulationNode>(".graph-node")
      .classed("is-active", (node) => active.has(node.paper_id))
      .classed(
        "is-edge-endpoint",
        (node) => hasHoveredEdge && active.has(node.paper_id),
      )
      .classed("is-selected", (node) => node.paper_id === selectedPaperId)
      .classed("is-selected-edge-endpoint", (node) => selectedIds.has(node.paper_id))
      .classed(
        "is-muted",
        (node) => hasActive && !active.has(node.paper_id) && !selectedIds.has(node.paper_id),
      );
    svg
      .selectAll<SVGLineElement, SimulationEdge>(".graph-edge")
      .classed("is-active", (link) => {
        const source = endpointId(link.source);
        const target = endpointId(link.target);
        if (hoveredEdge) {
          return (
            source === hoveredEdge.sourcePaperId && target === hoveredEdge.targetPaperId
          );
        }
        return active.has(source) || active.has(target);
      })
      .classed("is-hovered-edge", (link) => {
        if (!hoveredEdge) return false;
        return (
          endpointId(link.source) === hoveredEdge.sourcePaperId &&
          endpointId(link.target) === hoveredEdge.targetPaperId
        );
      })
      .classed("is-selected-edge", (link) => {
        if (!selectedEdge) return false;
        return (
          endpointId(link.source) === selectedEdge.source_paper_id &&
          endpointId(link.target) === selectedEdge.target_paper_id
        );
      })
      .classed("is-muted", (link) => {
        const source = endpointId(link.source);
        const target = endpointId(link.target);
        if (hoveredEdge) {
          if (
            selectedEdge &&
            source === selectedEdge.source_paper_id &&
            target === selectedEdge.target_paper_id
          ) return false;
          return !(
            source === hoveredEdge.sourcePaperId && target === hoveredEdge.targetPaperId
          );
        }
        return hasActive && !active.has(source) && !active.has(target);
      });
  }, [activePaperIds, hoveredEdge, selectedEdge, selectedPaperId]);

  return (
    <div className="research-graph" data-testid="research-graph">
      <svg ref={svgRef} aria-label="Paper semantic network" />
      {tooltip && (
        <div className="graph-tooltip" style={{ left: tooltip.x, top: tooltip.y }} role="tooltip">
          {tooltip.content}
        </div>
      )}
      <GraphLegend />
    </div>
  );
}

function GraphLegend() {
  return (
    <div className="graph-legend" aria-label="Graph legend">
      <div className="graph-legend-group" data-testid="node-size-legend">
        <b>Node size</b>
        <div className="graph-legend-scale">
          <span>Low</span>
          <span className="graph-legend-examples" aria-hidden="true">
            <i className="size-small" />
            <i className="size-medium" />
            <i className="size-large" />
          </span>
          <span>High</span>
        </div>
        <small>Query similarity</small>
      </div>
      <div className="graph-legend-group" data-testid="node-opacity-legend">
        <b>Node opacity</b>
        <div className="graph-legend-scale">
          <span>Low</span>
          <span className="graph-legend-examples" aria-hidden="true">
            <i className="opacity-low" />
            <i className="opacity-medium" />
            <i className="opacity-high" />
          </span>
          <span>High</span>
        </div>
        <small>Information completeness</small>
      </div>
      <div className="graph-legend-group" data-testid="edge-width-legend">
        <b>Edge width</b>
        <div className="graph-legend-scale">
          <span>Low</span>
          <span className="graph-legend-examples" aria-hidden="true">
            <i
              className="edge-thin"
              data-edge-weight={MIN_EDGE_SIMILARITY}
              style={{ height: `${edgeWidth(MIN_EDGE_SIMILARITY)}px` }}
            />
            <i
              className="edge-medium"
              data-edge-weight={0.65}
              style={{ height: `${edgeWidth(0.65)}px` }}
            />
            <i
              className="edge-thick"
              data-edge-weight={1}
              style={{ height: `${edgeWidth(1)}px` }}
            />
          </span>
          <span>High</span>
        </div>
        <small>Paper similarity</small>
      </div>
    </div>
  );
}
