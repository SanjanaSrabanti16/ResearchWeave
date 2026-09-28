import { useEffect, useState } from "react";

import {
  acquireAndParsePDF,
  APIError,
  extractPaperInsights,
  getLLMProviders,
  uploadAndParsePDF,
} from "../api/client";
import type {
  InsightsResponse,
  LLMProviderId,
  LLMProviderStatus,
  Paper,
  ParsedPaper,
} from "../types/paper";
import { PaperInsights } from "./PaperInsights";

interface Props {
  paper: Paper;
  initialDocument?: ParsedPaper | null;
  initialInsights?: InsightsResponse | null;
  initialProvider?: LLMProviderId | null;
  onStateChanged?: () => Promise<void> | void;
}

const PROVIDER_LABELS: Record<LLMProviderId, string> = {
  ollama: "Local Ollama",
  gemini: "Google Gemini",
  evl_gemma: "EVL Gemma",
};

const PROVIDER_DESCRIPTIONS: Record<LLMProviderId, string> = {
  ollama: "Local/private; quality and speed depend on your local model and hardware.",
  gemini:
    "Cloud analysis with stronger large-context understanding; sends selected paper text to Google.",
  evl_gemma:
    "Remote EVL-hosted model; requires configured EVL access and sends selected paper text to the EVL inference service.",
};

export function PaperPdfPanel({
  paper,
  initialDocument = null,
  initialInsights = null,
  initialProvider = null,
  onStateChanged,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [uploadRequired, setUploadRequired] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [document, setDocument] = useState<ParsedPaper | null>(initialDocument);
  const [message, setMessage] = useState<string | null>(null);
  const [insights, setInsights] = useState<InsightsResponse | null>(initialInsights);
  const [insightBusy, setInsightBusy] = useState(false);
  const [insightStage, setInsightStage] = useState("Selecting evidence");
  const [insightElapsed, setInsightElapsed] = useState(0);
  const [insightError, setInsightError] = useState<string | null>(null);
  const [provider, setProvider] = useState<LLMProviderId>(initialProvider ?? "ollama");
  const [providers, setProviders] = useState<LLMProviderStatus[]>([
    { provider_id: "ollama", model: "local model", configured: true, cloud: false },
    { provider_id: "gemini", model: "Gemini", configured: false, cloud: true },
    { provider_id: "evl_gemma", model: "Gemma", configured: false, cloud: true },
  ]);

  useEffect(() => {
    setDocument(initialDocument);
    setInsights(initialInsights);
    if (initialProvider) setProvider(initialProvider);
  }, [paper.id, initialDocument, initialInsights, initialProvider]);

  useEffect(() => {
    if (!document) return;
    let active = true;
    getLLMProviders()
      .then((result) => {
        if (!active) return;
        setProviders(result.providers);
        const cachedProvider = initialProvider
          ? result.providers.find(
              (item) => item.provider_id === initialProvider && item.configured,
            )
          : undefined;
        const preferred = result.providers.find(
          (item) => item.provider_id === result.default_provider && item.configured,
        );
        if (cachedProvider) setProvider(cachedProvider.provider_id);
        else if (preferred) setProvider(preferred.provider_id);
      })
      .catch(() => {
        // Ollama remains the safe backward-compatible selection if status is unavailable.
      });
    return () => {
      active = false;
    };
  }, [document, initialProvider]);

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
      if (result.document) await onStateChanged?.();
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
      if (result.document) await onStateChanged?.();
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
      const result = await extractPaperInsights(document, setInsightStage, provider);
      setInsights(result);
      await onStateChanged?.();
    } catch (error) {
      setInsightError(error instanceof Error ? error.message : "Insight extraction failed");
      await onStateChanged?.();
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
          <div className="insight-provider-control">
            <label htmlFor={`insight-provider-${paper.id}`}>Analysis provider</label>
            <select
              id={`insight-provider-${paper.id}`}
              value={provider}
              onChange={(event) => {
                setProvider(event.target.value as LLMProviderId);
                setInsights(null);
                setInsightError(null);
              }}
              disabled={insightBusy}
            >
              {providers.map((item) => (
                <option
                  key={item.provider_id}
                  value={item.provider_id}
                  disabled={!item.configured}
                >
                  {PROVIDER_LABELS[item.provider_id]}
                  {!item.configured ? " (unavailable)" : ""}
                </option>
              ))}
            </select>
            <p className="parser-provenance">
              {PROVIDER_DESCRIPTIONS[provider]}
            </p>
            <button
              type="button"
              className="secondary-button"
              onClick={extractInsights}
              disabled={insightBusy}
            >
              {insightBusy ? "Extracting insights..." : "Extract grounded insights"}
            </button>
          </div>
          {insightBusy && (
            <p className="pdf-message" role="status">
              {insightStage} ({insightElapsed}s elapsed)
            </p>
          )}
          {insightError && <p className="pdf-message" role="alert">{insightError}</p>}
          {insights && (
            <>
              <p className="parser-provenance">
                Grounded with {PROVIDER_LABELS[provider]} ({insights.model})
                {insights.cached ? " (cached)" : ""}.
              </p>
              <PaperInsights insights={insights.insights} />
            </>
          )}
        </div>
      )}
    </section>
  );
}
