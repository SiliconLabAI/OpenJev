import type { Answer, EvaluateResponse } from "../lib/types";

interface Props {
  result: EvaluateResponse | null;
  error: string | null;
  loading: boolean;
}

function ChoiceResult({ name, answer }: { name: string; answer: Extract<Answer, { type: "choice" }> }) {
  const entries = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="answer-card">
      <div className="answer-card-header">
        <span className="question-type-badge choice">choice</span>
        <span className="answer-name">{name}</span>
        <span className="answer-value" style={{ color: "var(--choice)" }}>{answer.choice}</span>
      </div>
      <div className="answer-body">
        {entries.map(([key, p]) => (
          <div className="prob-bar-row" key={key}>
            <span className="prob-label">{key}</span>
            <div className="prob-track">
              <div className="prob-fill choice" style={{ width: `${Math.round(p * 100)}%` }} />
            </div>
            <span className="prob-pct">{(p * 100).toFixed(0)}%</span>
          </div>
        ))}
        <div className="confidence-row">
          Confidence <span className="confidence-val">{(answer.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

function ScoreResult({ name, answer }: { name: string; answer: Extract<Answer, { type: "score" }> }) {
  const max = Math.max(0, ...Object.keys(answer.legend).map((k) => Number(k)));
  const label =
    answer.legend[String(Math.round(answer.score))] ??
    answer.legend[String(Math.floor(answer.score))] ??
    "";
  const probs =
    answer.probabilities ??
    Object.fromEntries(
      Object.keys(answer.legend).map((k) => [k, Number(k) === Math.round(answer.score) ? 1 : 0])
    );
  return (
    <div className="answer-card">
      <div className="answer-card-header">
        <span className="question-type-badge score">score</span>
        <span className="answer-name">{name}</span>
      </div>
      <div className="answer-body">
        <div className="score-gauge">
          <span className="score-number">{answer.score.toFixed(2)}</span>
          <span className="score-max">/ {max}</span>
        </div>
        {label && <div className="score-legend">{label}</div>}
        {Object.entries(answer.legend).map(([k, v]) => {
          const p = probs[k] ?? 0;
          return (
            <div className="prob-bar-row" key={k}>
              <span className="prob-label">{k}: {v}</span>
              <div className="prob-track">
                <div className="prob-fill score" style={{ width: `${Math.round(p * 100)}%` }} />
              </div>
              <span className="prob-pct">{(p * 100).toFixed(0)}%</span>
            </div>
          );
        })}
        <div className="confidence-row">
          Confidence <span className="confidence-val">{(answer.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

function NoulResult({ name, answer }: { name: string; answer: Extract<Answer, { type: "noul" }> }) {
  const pct = Math.round(answer.noul * 100);
  return (
    <div className="answer-card">
      <div className="answer-card-header">
        <span className="question-type-badge noul">noul</span>
        <span className="answer-name">{name}</span>
      </div>
      <div className="answer-body">
        <div className="noul-big">{pct}%</div>
        <div className="noul-label">probability true</div>
        <div className="prob-bar-row">
          <span className="prob-label">true</span>
          <div className="prob-track">
            <div className="prob-fill noul" style={{ width: `${pct}%` }} />
          </div>
          <span className="prob-pct">{pct}%</span>
        </div>
        <div className="prob-bar-row">
          <span className="prob-label">false</span>
          <div className="prob-track">
            <div className="prob-fill noul" style={{ width: `${100 - pct}%`, opacity: 0.45 }} />
          </div>
          <span className="prob-pct">{100 - pct}%</span>
        </div>
      </div>
    </div>
  );
}

export function ResultsPanel({ result, error, loading }: Props) {
  if (loading) {
    return (
      <div className="results-empty">
        <div className="spinner" style={{ width: 28, height: 28, borderWidth: 3 }} />
        <div>Running…</div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="panel-body">
        <div className="error-box">{error}</div>
      </div>
    );
  }
  if (!result) {
    return (
      <div className="results-empty">
        <div className="results-empty-icon">◎</div>
        <div style={{ fontWeight: 500, color: "var(--text-muted)" }}>OpenJev decision will appear here</div>
        <div style={{ fontSize: 13, maxWidth: 300 }}>
          Modes: parallel · oneshot · decider (Mapika/decider RLCD weights)
        </div>
      </div>
    );
  }
  return (
    <div className="panel-body">
      {Object.entries(result.answers).map(([name, answer]) => {
        if (answer.type === "choice") return <ChoiceResult key={name} name={name} answer={answer} />;
        if (answer.type === "score") return <ScoreResult key={name} name={name} answer={answer} />;
        return <NoulResult key={name} name={name} answer={answer} />;
      })}
      <div className="usage-footer">
        <span>model: {result.model}</span>
        {result.meta && (
          <>
            <span>mode: {result.meta.mode}</span>
            {result.meta.backend && <span>backend: {result.meta.backend}</span>}
            <span>calls: {result.meta.parallel_calls}</span>
            <span>{result.meta.latency_ms} ms</span>
          </>
        )}
        {result.usage.input_tokens != null && <span>in: {result.usage.input_tokens} tok</span>}
      </div>
    </div>
  );
}
