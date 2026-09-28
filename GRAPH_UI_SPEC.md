# ResearchWeave graph UI specification (M3.8–M3.13)

This specification freezes the first complete graph experience pending the user's M3.14 live
acceptance test. It does not change the graph semantics frozen in M3.1–M3.7.

## Workspace

- Compact ResearchWeave branding and the existing query, start year, end year, paper count, and
  Search controls occupy the top area.
- The remaining desktop viewport is split 58%/42%: SVG paper network on the left and a scrollable
  ranked-paper list on the right. Narrow layouts stack the panels.
- Search runs once. `/api/graph` receives the original query and those already-selected canonical
  papers; graph generation never reruns scholarly retrieval.
- A graph failure is non-fatal and leaves the paper list available.

## Frozen visual bindings

- SVG circle radius uses backend `node_radius` exactly.
- SVG circle base opacity uses backend `node_opacity` exactly.
- Nodes use one neutral teal hue; color has no analytical meaning.
- Only backend edges are rendered. Pixel width is `1 + edge_weight * 4`, giving a 1–5 px range.
- The force layout, collision, drag, zoom, and pan change geometry only—not semantic values.
- The bottom legend explains query relevance, information completeness, and paper similarity.

## Linked interaction

- Node hover shows title, query relevance, and information state; its paper row is highlighted.
- Paper-row hover highlights the corresponding node and incident edges.
- Edge hover highlights both endpoints and paper rows and shows both titles plus semantic
  similarity. It does not invent a relationship explanation.
- Clicking a node or row selects the same canonical paper and replaces the list with a right-side
  detail panel while preserving the mounted graph and its simulation state.
- Back restores the existing list without search or graph regeneration.

## Paper details

- The panel shows full metadata and abstract, then reuses the existing PDF acquisition, parsing,
  provider selection, insight extraction, `PaperInsights`, and expandable evidence components.
- A read-only cached-analysis lookup loads an existing parsed document and current grounded insight
  result by canonical paper ID. Clicking never starts LLM extraction automatically.
- Insight claims, including Paper Overview, are displayed in full without line clamps or ellipses;
  evidence remains collapsible.
