import React, { useState } from "react";
import { Newspaper, Send, RotateCcw, Sparkles, AlertCircle, Loader2, Info } from "lucide-react";
import { BiasCard } from "../components/BiasCard";
import { predictBias, fetchExplanation } from "../services/api";

const SAMPLE_TEXTS = [
  {
    label: "Strongly Biased Excerpt",
    text: "The corrupt politicians shamelessly betrayed ordinary hardworking citizens with their disastrous and reckless economic scheme."
  },
  {
    label: "Neutral Journalistic Excerpt",
    text: "The joint economic committee convened on Thursday to examine the quarterly manufacturing and employment metrics."
  },
  {
    label: "Mixed Multi-Sentence Article",
    text: "The municipal water department released its annual infrastructure inspection report on Tuesday. Engineers confirmed that three pump stations require scheduled maintenance before winter. However, critics argued that corrupt officials pushed through a disastrous contract giveaway. The city council agreed to allocate funds from the capital improvement reserve."
  }
];

export function BiasDetectionPage({ apiHealth }) {
  const [inputText, setInputText] = useState("");
  const [includeExplanation, setIncludeExplanation] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState(false);
  const [explainingSentenceIdx, setExplainingSentenceIdx] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [result, setResult] = useState(null);

  const handleAnalyze = async (e) => {
    if (e) e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed) {
      setErrorMessage("Please enter or paste news text to evaluate.");
      return;
    }

    setIsLoading(true);
    setErrorMessage("");

    try {
      const data = await predictBias(trimmed, includeExplanation);
      setResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setErrorMessage(err.message || "Failed to analyze text. Please verify the backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setInputText("");
    setResult(null);
    setErrorMessage("");
    setExplainingSentenceIdx(null);
  };

  const handleRequestTopExplanation = async () => {
    if (!result || !result.text) return;
    setIsLoadingExplanation(true);
    try {
      const exp = await fetchExplanation(result.text, 10);
      setResult((prev) => ({ ...prev, explanation: exp }));
    } catch (err) {
      console.error("Top explanation error:", err);
      alert(`Could not generate SHAP explanation: ${err.message}`);
    } finally {
      setIsLoadingExplanation(false);
    }
  };

  const handleExplainSentence = async (sentenceIndex, sentenceText) => {
    setExplainingSentenceIdx(sentenceIndex);
    try {
      const exp = await fetchExplanation(result.text, 10, sentenceIndex);
      setResult((prev) => {
        const updatedSentences = prev.sentences.map((s) =>
          s.index === sentenceIndex ? { ...s, explanation: exp } : s
        );
        return { ...prev, sentences: updatedSentences };
      });
    } catch (err) {
      console.error(`Sentence ${sentenceIndex} explanation error:`, err);
      alert(`Could not explain sentence ${sentenceIndex}: ${err.message}`);
    } finally {
      setExplainingSentenceIdx(null);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem" }}>
      {/* Input Card */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Newspaper size={20} color="#818cf8" />
            <h2 style={{ fontSize: "1.15rem", fontWeight: 700, margin: 0 }}>Input News Text</h2>
          </div>
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Evaluates across all 5 models & soft-voting ensemble
          </span>
        </div>

        {/* Sample text quick-fills */}
        <div className="sample-buttons">
          <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", alignSelf: "center", marginRight: "0.25rem" }}>
            Try a sample:
          </span>
          {SAMPLE_TEXTS.map((sample, idx) => (
            <button
              key={idx}
              type="button"
              className="sample-btn"
              onClick={() => {
                setInputText(sample.text);
                setErrorMessage("");
              }}
            >
              {sample.label}
            </button>
          ))}
        </div>

        <form onSubmit={handleAnalyze}>
          <div className="textarea-container">
            <textarea
              className="news-textarea"
              placeholder="Paste a news headline, paragraph, or article excerpt here to analyze for bias..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              rows={5}
              disabled={isLoading}
            />
            <div className="textarea-footer">
              <span>{inputText.length} characters</span>
              <span>Evaluates: Linear SVM, Logistic Regression, Naive Bayes, RoBERTa, DeBERTa</span>
            </div>
          </div>

          {errorMessage && (
            <div className="alert alert-error">
              <AlertCircle size={18} style={{ flexShrink: 0 }} />
              <div>{errorMessage}</div>
            </div>
          )}

          <div className="actions-row">
            <label className="toggle-label" style={{ cursor: "pointer", userSelect: "none" }}>
              <input
                type="checkbox"
                checked={includeExplanation}
                onChange={(e) => setIncludeExplanation(e.target.checked)}
                disabled={isLoading}
              />
              <Sparkles size={16} color="#818cf8" />
              <span>Compute SHAP Word Explanations (Takes ~2s on CPU)</span>
            </label>

            <div style={{ display: "flex", gap: "0.75rem" }}>
              {inputText && (
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={handleClear}
                  disabled={isLoading}
                >
                  <RotateCcw size={16} />
                  <span>Clear</span>
                </button>
              )}

              <button
                type="submit"
                className="analyze-btn"
                disabled={isLoading || !inputText.trim()}
              >
                {isLoading ? (
                  <>
                    <Loader2 size={18} className="spinner" />
                    <span>Evaluating 5 Models...</span>
                  </>
                ) : (
                  <>
                    <Send size={18} />
                    <span>Analyze Bias</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Model Scope & Prediction Notice */}
      <div
        style={{
          background: "rgba(15, 23, 42, 0.4)",
          border: "1px solid var(--border-color)",
          borderRadius: "0.5rem",
          padding: "0.75rem 1rem",
          fontSize: "0.82rem",
          color: "var(--text-muted)",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem"
        }}
      >
        <Info size={16} style={{ flexShrink: 0, color: "#818cf8" }} />
        <span>
          <strong>Scope Reminder:</strong> Predictions are statistical model estimates of lexical and framing bias, not definitive judgments about a news publisher or article.
        </span>
      </div>

      {/* Results Section */}
      {result && (
        <BiasCard
          result={result}
          onReset={handleClear}
          isLoadingExplanation={isLoadingExplanation}
          onRequestExplanation={handleRequestTopExplanation}
          onExplainSentence={handleExplainSentence}
          explainingSentenceIdx={explainingSentenceIdx}
        />
      )}
    </div>
  );
}
