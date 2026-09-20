import { useCallback, useState } from "react";
import {
  DEFAULT_EXAMPLES,
  draftsToQuestions,
  type EvaluateResponse,
  type QuestionDraft,
} from "./lib/types";
import { QuestionEditor, newQuestion } from "./components/QuestionEditor";
import { ResultsPanel } from "./components/ResultsPanel";

const STORAGE_KEY = "openjev-config";

function loadConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw)
      return JSON.parse(raw) as {
        model: string;
        baseUrl: string;
        apiKey: string;
        mode: "parallel" | "oneshot";
      };
  } catch {
    /* ignore */
  }
  return {
    model: "gpt-4o-mini",
    baseUrl: "",
    apiKey: "",
    mode: "parallel" as const,
  };
}

export function App() {
  const example0 = DEFAULT_EXAMPLES[0];
  const [activeExample, setActiveExample] = useState(example0.id);
  const [state, setState] = useState(example0.state);
  const [questions, setQuestions] = useState<QuestionDraft[]>(
    example0.questions
  );

  const [config, setConfig] = useState(loadConfig);
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const saveConfig = useCallback(
    (patch: Partial<typeof config>) => {
      const next = { ...config, ...patch };
      setConfig(next);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    },
    [config]
  );

  const loadExample = (id: string) => {
    const ex = DEFAULT_EXAMPLES.find((e) => e.id === id);
    if (!ex) return;
    setActiveExample(id);
    setState(ex.state);
    setQuestions(
      ex.questions.map((q) => ({ ...q, id: q.id + "-" + Date.now() }))
    );
    setResult(null);
    setError(null);
  };

  const updateQuestion = (id: string, q: QuestionDraft) => {
    setQuestions((prev) => prev.map((x) => (x.id === id ? q : x)));
  };

  const removeQuestion = (id: string) => {
    setQuestions((prev) => prev.filter((x) => x.id !== id));
  };

  const run = async () => {
    const qs = draftsToQuestions(questions);
    if (Object.keys(qs).length === 0) {
      setError("Add at least one valid question");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let parsedState: string | object = state;
      try {
        parsedState = JSON.parse(state);
      } catch {
        // plain string
      }

      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          state: parsedState,
          questions: qs,
          model: config.model || undefined,
          base_url: config.baseUrl || undefined,
          api_key: config.apiKey || undefined,
          temperature: 0,
          mode: config.mode,
        }),
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error || `HTTP ${res.status}`);
      } else {
        setResult(data as EvaluateResponse);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">OJ</div>
          <div>
            <div className="header-title">OpenJev</div>
            <div className="header-sub">
              Parallel System One decisions · open approximation of Jev
            </div>
          </div>
        </div>
        <div className="header-spacer" />
        <div className="header-config">
          <span className="config-label">mode</span>
          <select
            className="config-input"
            value={config.mode}
            onChange={(e) =>
              saveConfig({
                mode: e.target.value as "parallel" | "oneshot",
              })
            }
            style={{ minWidth: 110 }}
          >
            <option value="parallel">parallel</option>
            <option value="oneshot">oneshot</option>
          </select>
          <span className="config-label">model</span>
          <input
            className="config-input"
            value={config.model}
            onChange={(e) => saveConfig({ model: e.target.value })}
            placeholder="gpt-4o-mini"
            spellCheck={false}
          />
          <span className="config-label">base url</span>
          <input
            className="config-input wide"
            value={config.baseUrl}
            onChange={(e) => saveConfig({ baseUrl: e.target.value })}
            placeholder="https://api.openai.com/v1"
            spellCheck={false}
          />
          <span className="config-label">api key</span>
          <input
            className="config-input"
            type="password"
            value={config.apiKey}
            onChange={(e) => saveConfig({ apiKey: e.target.value })}
            placeholder="optional if set in .env"
            autoComplete="off"
          />
        </div>
      </header>

      <div className="main">
        <section className="panel">
          <div className="panel-header">
            <span className="panel-title">Input</span>
          </div>
          <div className="panel-body">
            <div className="examples">
              {DEFAULT_EXAMPLES.map((ex) => (
                <button
                  key={ex.id}
                  type="button"
                  className={`example-chip ${
                    activeExample === ex.id ? "active" : ""
                  }`}
                  onClick={() => loadExample(ex.id)}
                  title={ex.description}
                >
                  {ex.name}
                </button>
              ))}
            </div>

            <div className="field">
              <div className="field-label">
                State
                <span className="hint">{state.length} chars</span>
              </div>
              <textarea
                className="textarea"
                value={state}
                onChange={(e) => setState(e.target.value)}
                spellCheck={false}
                placeholder="Plain text or JSON object…"
              />
            </div>

            <div className="field">
              <div className="field-label">
                Questions
                <span className="hint">{questions.length}</span>
              </div>
              <div className="questions-list">
                {questions.map((q) => (
                  <QuestionEditor
                    key={q.id}
                    question={q}
                    onChange={(next) => updateQuestion(q.id, next)}
                    onRemove={() => removeQuestion(q.id)}
                  />
                ))}
              </div>
              <div className="add-question-row">
                <button
                  type="button"
                  className="add-type-btn"
                  onClick={() =>
                    setQuestions((p) => [...p, newQuestion("choice")])
                  }
                >
                  + Choice
                </button>
                <button
                  type="button"
                  className="add-type-btn"
                  onClick={() =>
                    setQuestions((p) => [...p, newQuestion("score")])
                  }
                >
                  + Score
                </button>
                <button
                  type="button"
                  className="add-type-btn"
                  onClick={() =>
                    setQuestions((p) => [...p, newQuestion("noul")])
                  }
                >
                  + Noul
                </button>
              </div>
            </div>
          </div>

          <div className="run-bar">
            <button
              type="button"
              className="run-btn"
              onClick={run}
              disabled={loading}
            >
              {loading && <span className="spinner" />}
              {loading ? "Sampling…" : "Run OpenJev"}
            </button>
            <span className="run-status">
              {config.mode === "parallel"
                ? "parallel sampler · each option scored independently"
                : "oneshot · single structured JSON call"}
            </span>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <span className="panel-title">Output</span>
            {result?.meta && (
              <span
                style={{
                  fontSize: 11,
                  color: "var(--text-dim)",
                  fontFamily: "var(--mono)",
                }}
              >
                {result.meta.mode} · {result.meta.parallel_calls} calls ·{" "}
                {result.meta.latency_ms} ms
              </span>
            )}
          </div>
          <ResultsPanel result={result} error={error} loading={loading} />
        </section>
      </div>
    </div>
  );
}
