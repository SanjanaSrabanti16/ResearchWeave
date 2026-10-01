import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  APIError,
  getPaperRelationshipByPair,
  getPaperRelationshipHistory,
  proposePaperRelationship,
  reviewPaperRelationship,
} from "../api/client";
import type {
  GraphEdge,
  Paper,
  RelationshipEvidence,
  RelationshipProposal,
  RelationshipReview,
  RelationshipReviewHistory,
  RelationshipType,
} from "../types/paper";

interface Props {
  edge: GraphEdge;
  sourcePaper: Paper;
  targetPaper: Paper;
  onBack: () => void;
  onOpenPaper: (paperId: string) => void;
}

const relationshipLabels: Record<RelationshipType, string> = {
  shared_problem: "Shared Problem",
  shared_method: "Shared Method",
  shared_finding: "Shared Finding",
  complementary_contribution: "Complementary Contribution",
  contrasting_result: "Contrasting Result",
  shared_limitation: "Shared Limitation",
  shared_future_work: "Shared Future Work",
  related_application: "Related Application",
  other: "Other Relationship",
};

const relationshipTypes = Object.keys(relationshipLabels) as RelationshipType[];
const missingAnalysisCodes = new Set([
  "PAPER_NOT_FOUND",
  "PARSED_PAPER_MISSING",
  "CURRENT_ANALYSIS_MISSING",
  "STALE_ANALYSIS",
]);

interface PanelError {
  kind: "missing" | "generation";
  message: string;
  retry: "lookup" | "generate" | null;
}

function shortTitle(title: string): string {
  return title.length <= 62 ? title : `${title.slice(0, 59)}…`;
}

function paperLabel(error: APIError, sourcePaper: Paper, targetPaper: Paper): string {
  if (error.message.includes(targetPaper.id)) return "Paper B";
  return "Paper A";
}

function missingMessage(error: APIError, sourcePaper: Paper, targetPaper: Paper): string {
  const label = paperLabel(error, sourcePaper, targetPaper);
  if (error.code === "PARSED_PAPER_MISSING") {
    return `${label} needs to be parsed before relationship analysis can run.`;
  }
  if (error.code === "CURRENT_ANALYSIS_MISSING") {
    return `${label} needs current structured insights before relationship analysis can run.`;
  }
  if (error.code === "STALE_ANALYSIS") {
    return `${label} needs refreshed structured insights before relationship analysis can run.`;
  }
  return `${label} is not available in the current backend paper state.`;
}

function relationshipError(caught: unknown, sourcePaper: Paper, targetPaper: Paper): PanelError {
  if (caught instanceof APIError && caught.code && missingAnalysisCodes.has(caught.code)) {
    return {
      kind: "missing",
      message: missingMessage(caught, sourcePaper, targetPaper),
      retry: null,
    };
  }
  return {
    kind: "generation",
    message: "Relationship analysis could not be generated.",
    retry: "lookup",
  };
}

function reviewLabel(review: RelationshipReview): string {
  if (review.decision === "accepted") return "Accepted";
  if (review.decision === "rejected") return "Rejected";
  return "Edited";
}

function pageLabel(evidence: RelationshipEvidence): string | null {
  if (evidence.page_start === null) return null;
  return evidence.page_end && evidence.page_end !== evidence.page_start
    ? `Pages ${evidence.page_start}–${evidence.page_end}`
    : `Page ${evidence.page_start}`;
}

function EvidenceGroup({
  title,
  evidence,
  evidenceIds,
}: {
  title: string;
  evidence: RelationshipEvidence[];
  evidenceIds: string[];
}) {
  const used = new Set(evidenceIds);
  const items = evidence.filter((item) => used.has(item.evidence_id));
  return (
    <details className="relationship-evidence" open>
      <summary>Evidence from {shortTitle(title)} ({items.length})</summary>
      <div className="relationship-evidence-list">
        {items.map((item, index) => (
          <article key={`${item.evidence_id}-${item.insight_field}-${index}`}>
            <p className="relationship-evidence-claim">{item.claim}</p>
            <blockquote>“{item.quote}”</blockquote>
            <p className="relationship-evidence-meta">
              {item.section_heading && <span>Section: {item.section_heading}</span>}
              {pageLabel(item) && <span>{pageLabel(item)}</span>}
              <code>{item.evidence_id}</code>
            </p>
          </article>
        ))}
      </div>
    </details>
  );
}

export function RelationshipDetail({
  edge,
  sourcePaper,
  targetPaper,
  onBack,
  onOpenPaper,
}: Props) {
  const [proposal, setProposal] = useState<RelationshipProposal | null>(null);
  const [history, setHistory] = useState<RelationshipReviewHistory | null>(null);
  const [checkingCache, setCheckingCache] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [panelError, setPanelError] = useState<PanelError | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [editedSummary, setEditedSummary] = useState("");
  const [editedTypes, setEditedTypes] = useState<RelationshipType[]>([]);
  const [reviewerNote, setReviewerNote] = useState("");
  const requestGeneration = useRef(0);
  const pendingAction = useRef(false);

  const latestReview = history?.reviews.at(-1) ?? null;
  const sourceTitle = sourcePaper.title;
  const targetTitle = targetPaper.title;

  const loadHistory = useCallback(async (current: RelationshipProposal, generation: number) => {
    try {
      const result = await getPaperRelationshipHistory(current.proposal_id);
      if (requestGeneration.current === generation) {
        setHistory(result);
        setHistoryError(null);
      }
    } catch {
      if (requestGeneration.current === generation) {
        setHistoryError("Review history is temporarily unavailable.");
      }
    }
  }, []);

  const refreshPair = useCallback(async (showChecking = true) => {
    const generation = ++requestGeneration.current;
    if (showChecking) setCheckingCache(true);
    setPanelError(null);
    try {
      const current = await getPaperRelationshipByPair(sourcePaper.id, targetPaper.id);
      if (requestGeneration.current !== generation) return;
      setProposal(current);
      setHistory(null);
      if (current) await loadHistory(current, generation);
    } catch (caught) {
      if (requestGeneration.current === generation) {
        setProposal(null);
        setHistory(null);
        setPanelError(relationshipError(caught, sourcePaper, targetPaper));
      }
    } finally {
      if (showChecking && requestGeneration.current === generation) setCheckingCache(false);
    }
  }, [loadHistory, sourcePaper, targetPaper]);

  useEffect(() => {
    setProposal(null);
    setHistory(null);
    setPanelError(null);
    setHistoryError(null);
    setEditing(false);
    void refreshPair(true);
    return () => {
      requestGeneration.current += 1;
    };
  }, [refreshPair]);

  async function generateRelationship() {
    if (pendingAction.current) return;
    pendingAction.current = true;
    setGenerating(true);
    setPanelError(null);
    try {
      await proposePaperRelationship({
        source_paper_id: sourcePaper.id,
        target_paper_id: targetPaper.id,
      });
      await refreshPair(false);
    } catch (caught) {
      const error = relationshipError(caught, sourcePaper, targetPaper);
      setPanelError({
        ...error,
        retry: error.kind === "generation" ? "generate" : null,
      });
    } finally {
      pendingAction.current = false;
      setGenerating(false);
    }
  }

  async function submitReview(
    decision: "accepted" | "rejected" | "edited",
    edited?: { types: RelationshipType[]; summary: string; note: string },
  ) {
    if (!proposal || pendingAction.current) return;
    pendingAction.current = true;
    setReviewing(true);
    setPanelError(null);
    try {
      await reviewPaperRelationship(proposal.proposal_id, {
        decision,
        ...(edited
          ? {
              edited_relationship_types: edited.types,
              edited_summary: edited.summary,
              ...(edited.note ? { reviewer_note: edited.note } : {}),
            }
          : {}),
      });
      setEditing(false);
      await refreshPair(false);
    } catch {
      setPanelError({
        kind: "generation",
        message: "The review could not be saved. The original proposal is unchanged.",
        retry: null,
      });
    } finally {
      pendingAction.current = false;
      setReviewing(false);
    }
  }

  function startEditing() {
    if (!proposal) return;
    setEditedSummary(latestReview?.edited_summary ?? proposal.summary);
    setEditedTypes(latestReview?.edited_relationship_types ?? proposal.relationship_types);
    setReviewerNote("");
    setEditing(true);
  }

  function toggleType(type: RelationshipType) {
    setEditedTypes((current) =>
      current.includes(type) ? current.filter((item) => item !== type) : [...current, type],
    );
  }

  function saveEdit(event: FormEvent) {
    event.preventDefault();
    if (!editedTypes.length || !editedSummary.trim()) return;
    void submitReview("edited", {
      types: editedTypes,
      summary: editedSummary.trim(),
      note: reviewerNote.trim(),
    });
  }

  const relationshipCards = useMemo(() => proposal?.relationships ?? [], [proposal]);

  return (
    <aside className="relationship-detail" aria-label={`Relationship between ${sourceTitle} and ${targetTitle}`}>
      <header className="relationship-detail-header">
        <button type="button" className="back-button" onClick={onBack}>← Back</button>
        <span className="relationship-panel-label">Relationship</span>
      </header>
      <div className="relationship-detail-scroll">
        <div className="relationship-paper-pair">
          <h2>{sourceTitle}</h2>
          <span aria-hidden="true">↕</span>
          <h2>{targetTitle}</h2>
          <p>Semantic similarity: <strong>{edge.paper_similarity.toFixed(2)}</strong></p>
        </div>

        {checkingCache && (
          <p className="relationship-panel-status" role="status">Checking existing relationship analysis…</p>
        )}

        {!checkingCache && panelError?.kind === "missing" && (
          <section className="relationship-state-card" role="status">
            <h3>Relationship Analysis</h3>
            <p>{panelError.message}</p>
            <div className="relationship-state-actions">
              <button type="button" className="secondary-button" onClick={() => onOpenPaper(sourcePaper.id)}>
                Open Paper A details
              </button>
              <button type="button" className="secondary-button" onClick={() => onOpenPaper(targetPaper.id)}>
                Open Paper B details
              </button>
            </div>
          </section>
        )}

        {!checkingCache && panelError?.kind === "generation" && (
          <section className="relationship-state-card relationship-error" role="alert">
            <h3>Relationship Analysis</h3>
            <p>{panelError.message}</p>
            {panelError.retry && (
              <button
                type="button"
                className="secondary-button"
                disabled={generating}
                onClick={() => {
                  if (panelError.retry === "generate") void generateRelationship();
                  else void refreshPair(true);
                }}
              >
                Retry
              </button>
            )}
          </section>
        )}

        {!checkingCache && !proposal && !panelError && !generating && (
          <section className="relationship-state-card">
            <h3>Relationship Analysis</h3>
            <p>
              ResearchWeave can compare the validated evidence from both papers to explain why
              they may be related.
            </p>
            <button type="button" onClick={() => void generateRelationship()}>
              Generate relationship analysis
            </button>
          </section>
        )}

        {generating && (
          <p className="relationship-panel-status" role="status">
            Analyzing evidence from both papers…
          </p>
        )}

        {proposal && !checkingCache && (
          <section className="relationship-ready" aria-label="Relationship analysis">
            <div className="relationship-ready-heading">
              <div>
                <h3>Relationship Analysis</h3>
                <p className="relationship-cache-label">
                  {proposal.cached ? "Cached evidence-grounded proposal" : "Evidence-grounded proposal"}
                </p>
              </div>
              {latestReview && (
                <span className={`relationship-review-status is-${latestReview.decision}`}>
                  {reviewLabel(latestReview)}
                </span>
              )}
            </div>

            <p className="relationship-proposal-summary">{proposal.summary}</p>

            {latestReview?.decision === "edited" && latestReview.edited_summary && (
              <section className="relationship-human-edit" aria-label="Human-edited interpretation">
                <h4>Human-edited interpretation</h4>
                <p>{latestReview.edited_summary}</p>
                <p className="relationship-type-list">
                  {latestReview.edited_relationship_types?.map((type) => relationshipLabels[type]).join(" · ")}
                </p>
              </section>
            )}

            <div className="relationship-card-list">
              {relationshipCards.map((item, index) => (
                <article className="relationship-card" key={`${item.relationship_type}-${index}`}>
                  <span className="relationship-type-badge">
                    {relationshipLabels[item.relationship_type]}
                  </span>
                  <p className="relationship-item-summary">{item.summary}</p>
                  <p className="relationship-confidence">
                    Confidence: {item.confidence.toFixed(2)}
                  </p>
                  <EvidenceGroup
                    title={sourceTitle}
                    evidence={proposal.evidence_source}
                    evidenceIds={item.source_evidence_ids}
                  />
                  <EvidenceGroup
                    title={targetTitle}
                    evidence={proposal.evidence_target}
                    evidenceIds={item.target_evidence_ids}
                  />
                </article>
              ))}
            </div>

            {!editing && (
              <div className="relationship-review-actions" aria-label="Review relationship proposal">
                <button
                  type="button"
                  disabled={reviewing}
                  onClick={() => void submitReview("accepted")}
                >
                  Accept
                </button>
                <button type="button" className="secondary-button" disabled={reviewing} onClick={startEditing}>
                  Edit
                </button>
                <button
                  type="button"
                  className="secondary-button relationship-reject-button"
                  disabled={reviewing}
                  onClick={() => void submitReview("rejected")}
                >
                  Reject
                </button>
              </div>
            )}

            {editing && (
              <form className="relationship-edit-form" onSubmit={saveEdit}>
                <fieldset>
                  <legend>Relationship types</legend>
                  <div className="relationship-type-options">
                    {relationshipTypes.map((type) => (
                      <label key={type}>
                        <input
                          type="checkbox"
                          checked={editedTypes.includes(type)}
                          onChange={() => toggleType(type)}
                        />
                        {relationshipLabels[type]}
                      </label>
                    ))}
                  </div>
                </fieldset>
                <label>
                  Relationship summary
                  <textarea
                    value={editedSummary}
                    onChange={(event) => setEditedSummary(event.target.value)}
                    minLength={3}
                    maxLength={2400}
                    required
                  />
                </label>
                <label>
                  Reviewer note (optional)
                  <textarea
                    value={reviewerNote}
                    onChange={(event) => setReviewerNote(event.target.value)}
                    maxLength={2000}
                  />
                </label>
                <div className="relationship-review-actions">
                  <button type="submit" disabled={reviewing || editedTypes.length === 0}>Save edit</button>
                  <button type="button" className="secondary-button" disabled={reviewing} onClick={() => setEditing(false)}>
                    Cancel
                  </button>
                </div>
              </form>
            )}

            {reviewing && <p className="relationship-panel-status" role="status">Saving review…</p>}
            {historyError && <p className="relationship-inline-warning" role="status">{historyError}</p>}
            {history && history.reviews.length > 0 && (
              <details className="relationship-history">
                <summary>Review history ({history.reviews.length})</summary>
                <ol>
                  {history.reviews.map((review) => (
                    <li key={review.review_id}>
                      <strong>{reviewLabel(review)}</strong>
                      <time dateTime={review.created_at}>
                        {new Date(review.created_at).toLocaleString()}
                      </time>
                      {review.edited_relationship_types && (
                        <p>{review.edited_relationship_types.map((type) => relationshipLabels[type]).join(" · ")}</p>
                      )}
                      {review.edited_summary && <p>{review.edited_summary}</p>}
                      {review.reviewer_note && <p>Note: {review.reviewer_note}</p>}
                    </li>
                  ))}
                </ol>
              </details>
            )}
          </section>
        )}
      </div>
    </aside>
  );
}
