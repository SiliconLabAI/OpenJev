/**
 * Deterministic option shuffling for position-bias robustness checks.
 * Kept in lib (not scripts) so it can be imported without side effects.
 */
import type { Questions } from "./types";

/** Deterministic xorshift32 Fisher–Yates shuffle (reproducible across runs). */
export function shuffleDeterministic<T>(items: T[], seed: number): T[] {
  let s = seed || 0x9e3779b9;
  const rand = () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0x100000000;
  };
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

/** Shuffle the criteria (option order) of every choice question, keeping keys attached to descriptions. */
export function shuffleChoiceCriteria(questions: Questions, seed = 42): Questions {
  const out: Questions = {};
  for (const [name, q] of Object.entries(questions)) {
    if (q.type === "choice") {
      const entries = shuffleDeterministic(Object.entries(q.criteria), seed);
      out[name] = { ...q, criteria: Object.fromEntries(entries) };
    } else {
      out[name] = q;
    }
  }
  return out;
}
