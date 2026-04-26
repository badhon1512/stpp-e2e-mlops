"use client";

import { FormEvent, useMemo, useState } from "react";

type PredictionResult = {
  predicted_priority: "low" | "medium" | "high" | string;
  confidence: number;
  model_uri: string;
  latency_ms: number;
};

const examples = [
  {
    label: "High",
    subject: "Production payment outage",
    body: "Customers cannot complete checkout and payment processing is down after the latest deploy."
  },
  {
    label: "Medium",
    subject: "Dashboard error after login",
    body: "The analytics dashboard shows an error for several users, but core workflows still work."
  },
  {
    label: "Low",
    subject: "Question about invoice export",
    body: "Can you confirm where I can download last month's invoice export?"
  }
];

function priorityClass(priority: string) {
  const normalized = priority.toLowerCase();
  if (normalized === "high") return "priority priorityHigh";
  if (normalized === "medium") return "priority priorityMedium";
  return "priority priorityLow";
}

export default function Home() {
  const [subject, setSubject] = useState(examples[0].subject);
  const [body, setBody] = useState(examples[0].body);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const confidencePercent = useMemo(() => {
    if (!result) return 0;
    return Math.round(result.confidence * 1000) / 10;
  }, [result]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject, body })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? `Prediction failed with status ${response.status}`);
      }

      setResult(data);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  function useExample(index: number) {
    setSubject(examples[index].subject);
    setBody(examples[index].body);
    setResult(null);
    setError(null);
  }

  return (
    <main className="shell">
      <section className="workspaceHeader">
        <div>
          <p className="eyebrow">Support operations</p>
          <h1>Ticket Priority Console</h1>
        </div>
        <div className="statusStrip" aria-label="Model status">
          <span className="statusDot" />
          FastAPI backend expected on *
        </div>
      </section>

      <section className="workspace">
        <form className="panel ticketPanel" onSubmit={handleSubmit}>
          <div className="panelHeader">
            <div>
              <h2>New Ticket</h2>
              <p>Submit subject and body text to the local prediction API.</p>
            </div>
          </div>

          <label className="field">
            <span>Subject</span>
            <input
              value={subject}
              onChange={(event) => setSubject(event.target.value)}
              minLength={3}
              placeholder="Production payment outage"
              required
            />
          </label>

          <label className="field">
            <span>Body</span>
            <textarea
              value={body}
              onChange={(event) => setBody(event.target.value)}
              minLength={5}
              rows={10}
              placeholder="Describe the issue, affected users, and urgency."
              required
            />
          </label>

          <div className="exampleRow" aria-label="Example tickets">
            {examples.map((example, index) => (
              <button key={example.label} type="button" onClick={() => useExample(index)}>
                {example.label}
              </button>
            ))}
          </div>

          <button className="submitButton" type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict Priority"}
          </button>
        </form>

        <aside className="panel resultPanel">
          <div className="panelHeader">
            <div>
              <h2>Prediction</h2>
              <p>Response from the active model.</p>
            </div>
          </div>

          {!result && !error && (
            <div className="emptyState">
              <strong>No prediction yet</strong>
              <span>Run a ticket through the model to see priority, confidence, and latency.</span>
            </div>
          )}

          {error && (
            <div className="errorBox">
              <strong>Request failed</strong>
              <span>{error}</span>
            </div>
          )}

          {result && (
            <div className="resultStack">
              <div className={priorityClass(result.predicted_priority)}>
                <span>Priority</span>
                <strong>{result.predicted_priority}</strong>
              </div>

              <div className="metricGrid">
                <div>
                  <span>Confidence</span>
                  <strong>{confidencePercent.toFixed(1)}%</strong>
                </div>
                <div>
                  <span>Latency</span>
                  <strong>{result.latency_ms} ms</strong>
                </div>
              </div>

              <div className="modelBlock">
                <span>Model URI</span>
                <code>{result.model_uri}</code>
              </div>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}
