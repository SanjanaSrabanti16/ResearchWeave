import { useEffect, useState } from "react";

import { acquireAndParsePDF, APIError, extractPaperInsights, uploadAndParsePDF } from "../api/client";
import type { InsightsResponse, Paper, ParsedPaper } from "../types/paper";
import { PaperInsights } from "./PaperInsights";

interface Props {
  paper: Paper;
}

export function PaperPdfPanel({ paper }: Props) {
  const [busy, setBusy] = useState(false);
  const [uploadRequired, setUploadRequired] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [document, setDocument] = useState<ParsedPaper | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [insights, setInsights] = useState<InsightsResponse | null>(null);
  const [insightBusy, setInsightBusy] = useState(false);
  const [insightStage, setInsightStage] = useState("Selecting evidence");
  const [insightElapsed, setInsightElapsed] = useState(0);
  const [insightError, setInsightError] = useState<string | null>(null);

  useEffect(() => {
    if (!insightBusy) return;
    const started = Date.now();
    const timer = window.setInterval(() => {
      setInsightElapsed(Math.floor((Date.now() - started) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [insightBusy]);

  async function acquire() {
    setBusy(true);
    setMessage("Looking for a validated open-access PDF...");
    try {
      const result = await acquireAndParsePDF(paper);
      setUploadRequired(result.status === "upload_required");
      setDocument(result.document);
      setInsights(null);
      setInsightError(null);
      setMessage(result.message);
    } catch (error) {
      setUploadRequired(!(error instanceof APIError && error.status === 503));
      setMessage(error instanceof Error ? error.message : "PDF acquisition failed");
    } finally {
      setBusy(false);
    }
  }

  async function upload() {
    if (!file) return;
    setBusy(true);
    setMessage("Validating and parsing the uploaded PDF...");
    try {
      const result = await uploadAndParsePDF(paper.id, file);
      setDocument(result.document);
      setInsights(null);
      setInsightError(null);
      setUploadRequired(false);
      setMessage(result.message);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "PDF upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function extractInsights() {
    if (!document) return;
    setInsightBusy(true);
    setInsightStage("Selecting evidence");
    setInsightElapsed(0);
    setInsightError(null);
    try {
      setInsights(await extractPaperInsights(document, setInsightStage));
    } catch (error) {
      setInsightError(error instanceof Error ? error.message : "Insight extraction failed");
    } finally {
      setInsightBusy(false);
    }
  }

  return (
    <section className="pdf-panel" aria-label="PDF processing">
      {!document && (
        <button type="button" className="secondary-button" onClick={acquire} disabled={busy}>
          {busy ? "Processing PDF..." : "Get PDF"}
        </button>
      )}
      {message && <p className={document ? "pdf-message success" : "pdf-message"}>{message}</p>}
      {uploadRequired && !document && (
        <div className="upload-controls">
          <label>
            Upload a PDF you are authorized to use
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              disabled={busy}
            />
          </label>
          <button type="button" onClick={upload} disabled={busy || !file}>
            Upload and parse
          </button>
        </div>
      )}
      {document && (
        <div className="document-preview">
          <h3>{document.title ?? paper.title}</h3>
          <p className="metadata">
            {document.authors.length ? document.authors.join(", ") : "Authors unavailable"}
          </p>
          <h4>Abstract</h4>
          <p>{document.abstract ?? "No abstract extracted."}</p>
          <h4>Main sections</h4>
          {document.sections.slice(0, 6).map((section) => (
            <section key={section.id}>
              <h5>{section.heading ?? "Untitled section"}</h5>
              <p>{section.text || "No paragraph text extracted."}</p>
            </section>
          ))}
          <p className="reference-count">References extracted: {document.references.length}</p>
          <p className="parser-provenance">
            Parsed by {document.parser} {document.parser_version} from a validated{" "}
            {document.source_pdf.acquisition_method.replace("_", " ")} PDF.
          </p>
          <button
            type="button"
            className="secondary-button"
            onClick={extractInsights}
            disabled={insightBusy}
          >
            {insightBusy ? "Extracting insights..." : "Extract grounded insights"}
          </button>
          {insightBusy && (
            <p className="pdf-message" role="status">
              {insightStage} ({insightElapsed}s elapsed)
            </p>
          )}
          {insightError && <p className="pdf-message" role="alert">{insightError}</p>}
          {insights && (
            <>
              <p className="parser-provenance">
                Grounded with {insights.model}{insights.cached ? " (cached)" : ""}.
              </p>
              <PaperInsights insights={insights.insights} />
            </>
          )}
        </div>
      )}
    </section>
  );
}
