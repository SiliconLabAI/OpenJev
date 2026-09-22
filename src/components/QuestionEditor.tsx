import type { QuestionDraft, QuestionType } from "../lib/types";

interface Props {
  question: QuestionDraft;
  onChange: (q: QuestionDraft) => void;
  onRemove: () => void;
}

export function QuestionEditor({ question, onChange, onRemove }: Props) {
  const update = (patch: Partial<QuestionDraft>) => onChange({ ...question, ...patch });
  const typeLabel = question.type === "boolean" ? "noul" : question.type;

  return (
    <div className="question-card">
      <div className="question-card-header">
        <span className={`question-type-badge ${typeLabel}`}>{typeLabel}</span>
        <input
          className="question-name-input"
          value={question.name}
          onChange={(e) => update({ name: e.target.value })}
          placeholder="question_name"
          spellCheck={false}
        />
        <button type="button" className="question-remove" onClick={onRemove} aria-label="Remove">
          ✕
        </button>
      </div>
      <div className="question-card-body">
        <input
          className="question-instructions"
          value={question.instructions}
          onChange={(e) => update({ instructions: e.target.value })}
          placeholder="Instructions for the model…"
        />
        {question.type === "choice" && (
          <div className="options-list">
            {question.options.map((opt, i) => (
              <div className="option-row" key={i}>
                <input
                  value={opt.key}
                  onChange={(e) => {
                    const options = [...question.options];
                    options[i] = { ...opt, key: e.target.value };
                    update({ options });
                  }}
                  placeholder="key"
                  spellCheck={false}
                />
                <input
                  value={opt.description}
                  onChange={(e) => {
                    const options = [...question.options];
                    options[i] = { ...opt, description: e.target.value };
                    update({ options });
                  }}
                  placeholder="description"
                />
                <button
                  type="button"
                  className="option-remove"
                  onClick={() => update({ options: question.options.filter((_, j) => j !== i) })}
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              type="button"
              className="add-option-btn"
              onClick={() => update({ options: [...question.options, { key: "", description: "" }] })}
            >
              + Add option
            </button>
          </div>
        )}
        {question.type === "score" && (
          <div className="options-list">
            {question.levels.map((lvl, i) => (
              <div className="option-row" key={i}>
                <input value={String(i)} disabled style={{ opacity: 0.5, width: 40 }} />
                <input
                  value={lvl}
                  onChange={(e) => {
                    const levels = [...question.levels];
                    levels[i] = e.target.value;
                    update({ levels });
                  }}
                  placeholder={`Level ${i} label`}
                />
                <button
                  type="button"
                  className="option-remove"
                  onClick={() => update({ levels: question.levels.filter((_, j) => j !== i) })}
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              type="button"
              className="add-option-btn"
              onClick={() => update({ levels: [...question.levels, ""] })}
            >
              + Add level
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export function newQuestion(type: QuestionType): QuestionDraft {
  const id = `q-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  if (type === "choice") {
    return {
      id,
      name: "choice_q",
      type: "choice",
      instructions: "",
      options: [
        { key: "a", description: "" },
        { key: "b", description: "" },
      ],
      levels: [],
    };
  }
  if (type === "score") {
    return {
      id,
      name: "score_q",
      type: "score",
      instructions: "",
      options: [],
      levels: ["Low", "Medium", "High"],
    };
  }
  return { id, name: "noul_q", type: "noul", instructions: "", options: [], levels: [] };
}
