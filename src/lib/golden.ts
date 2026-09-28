/**
 * Golden-set evaluation harness.
 *
 * A "golden set" is a JSONL file of labeled evaluation rows: each row carries
 * the request (state + questions) and expectations per question. The harness
 * runs the rows through evaluate(), scores accuracy plus calibration (Brier,
 * ECE), and compares a run against a stored baseline so CI can fail on
 * regressions.
 *
 * Expectation shapes per question type:
 *   choice: { "choice": "<option key>" }
 *   score:  { "score": <number> }            (scored against expected value)
 *   noul:   { "noul": <0..1> }
 *
 * Rows missing an expectation for a question are skipped for that question.
 */

import { brierScore, expectedCalibrationError, type FitSample } from "./calibration";
import type { Answer, Questions } from "./types";

// ── dataset shape ───────────────────────────────────────────────────────────

export type Expectation = { choice: string } | { score: number } | { noul: number };

export interface GoldenRow {
  name?: string;
  state: EvaluateRequestLike["state"];
  questions: Questions;
  expectations: Record<string, Expectation>;
  /** Optional per-row overrides (model, calibration_temperature, ...). */
  request?: Partial<EvaluateRequestLike>;
}

/** Structural subset of EvaluateRequest — keeps this module testable without the SDK. */
export interface EvaluateRequestLike {
  state: string | Record<string, unknown> | unknown[];
  questions: Questions;
  model?: string;
  base_url?: string;
  api_key?: string;
  temperature?: number;
  mode?: "parallel" | "oneshot" | "decider";
  calibration_temperature?: number;
  readout?: "logprob" | "micro";
}

/** Structural subset of EvaluateResponse. */
export interface EvaluateResponseLike {
  answers: Record<string, Answer>;
}

export function parseGoldenRow(raw: unknown, line: number, path: string): GoldenRow {
  const row = raw as GoldenRow;
  if (!row || typeof row !== "object") {
    throw new Error(`${path}:${line}: row must be an object`);
  }
  if (row.state === undefined || row.state === null) {
    throw new Error(`${path}:${line}: missing "state"`);
  }
  if (!row.questions || typeof row.questions !== "object") {
    throw new Error(`${path}:${line}: missing "questions"`);
  }
  if (!row.expectations || typeof row.expectations !== "object") {
    throw new Error(`${path}:${line}: missing "expectations"`);
  }
  return row;
}

// ── expectation → probability vector ───────────────────────────────────────

/** Which probability vector an answer exposes, normalized to its keys order. */
export function answerProbs(answer: Answer): { keys: string[]; probs: number[] } | null {
  if (answer.type === "choice" && answer.probabilities) {
    const keys = Object.keys(answer.probabilities);
    return { keys, probs: keys.map((k) => answer.probabilities![k] ?? 0) };
  }
  if (answer.type === "score" && answer.probabilities) {
    const keys = Object.keys(answer.probabilities);
    return { keys, probs: keys.map((k) => answer.probabilities![k] ?? 0) };
  }
  return null;
}

/**
 * Convert an answer + expectation into a FitSample (probs + correct index).
 * Returns null when the expectation can't be scored against this answer
 * (missing probabilities, unknown choice key, out-of-range numbers).
 */
export function toFitSample(
  answer: Answer,
  expectation: Expectation
): FitSample | null {
  if ("choice" in expectation) {
    const spec = answerProbs(answer);
    if (!spec) return null;
    const idx = spec.keys.indexOf(expectation.choice);
    if (idx === -1) return null;
    return { probs: spec.probs, correctIdx: idx };
  }
  if ("score" in expectation) {
    // Score answers distribute over levels; the expectation is the target
    // level index (rounded), scored as a classification over the levels.
    const spec = answerProbs(answer);
    if (!spec) return null;
    const idx = Math.round(expectation.score);
    if (idx < 0 || idx >= spec.probs.length) return null;
    return { probs: spec.probs, correctIdx: idx };
  }
  if ("noul" in expectation) {
    // Binary event: [yes, no]. The noul answer IS P(yes).
    if (answer.type !== "noul") return null;
    const noul = answer.noul;
    if (typeof noul !== "number" || !Number.isFinite(noul)) return null;
    const p = Math.min(1, Math.max(0, noul));
    const yes = expectation.noul >= 0.5;
    return { probs: [p, 1 - p], correctIdx: yes ? 0 : 1 };
  }
  return null;
}

// ── aggregation ────────────────────────────────────────────────────────────

export interface GoldenMetrics {
  questions_total: number;
  questions_scored: number;
  questions_skipped: number;
  /** Fraction where argmax(probs) matches the expected choice/level. */
  accuracy: number;
  /** Mean multiclass Brier over scored questions (0 = perfect, 2 = worst). */
  brier: number;
  /** Expected calibration error over scored questions (0 = perfect). */
  ece: number;
}

export function aggregateMetrics(
  samples: FitSample[]
): GoldenMetrics {
  const n = samples.length;
  const correct = samples.filter((s) => argmaxLocal(s.probs) === s.correctIdx).length;
  const brier = n
    ? samples.reduce((acc, s) => acc + brierScore(s.probs, s.correctIdx), 0) / n
    : 0;
  return {
    questions_total: 0, // filled in by the caller
    questions_scored: n,
    questions_skipped: 0, // filled in by the caller
    accuracy: n ? correct / n : 0,
    brier,
    ece: n ? expectedCalibrationError(samples) : 0,
  };
}

function argmaxLocal(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function scoreResponse(
  row: GoldenRow,
  response: EvaluateResponseLike
): { samples: FitSample[]; scored: number; skipped: number } {
  const samples: FitSample[] = [];
  let skipped = 0;
  for (const [name, expectation] of Object.entries(row.expectations)) {
    const answer = response.answers[name];
    if (!answer) {
      skipped++;
      continue;
    }
    const sample = toFitSample(answer, expectation);
    if (sample) samples.push(sample);
    else skipped++;
  }
  return { samples, scored: samples.length, skipped };
}

// ── baseline comparison ────────────────────────────────────────────────────

export interface GoldenBaseline {
  model?: string;
  created_at?: string;
  dataset_path?: string;
  metrics: Omit<GoldenMetrics, "questions_scored" | "questions_skipped"> & {
    questions_scored: number;
  };
}

export interface RegressionCheck {
  ok: boolean;
  failures: string[];
  deltas: Record<string, number>;
}

export const REGRESSION_THRESHOLDS = {
  /** Max allowed increase in mean Brier score. */
  brier: 0.02,
  /** Max allowed increase in ECE. */
  ece: 0.02,
  /** Min required accuracy (fraction, not percentage). */
  accuracy_drop: 0.02,
};

export function compareAgainstBaseline(
  current: Pick<GoldenMetrics, "accuracy" | "brier" | "ece">,
  baseline: GoldenBaseline,
  thresholds: typeof REGRESSION_THRESHOLDS = REGRESSION_THRESHOLDS
): RegressionCheck {
  const deltas = {
    accuracy: current.accuracy - baseline.metrics.accuracy,
    brier: current.brier - baseline.metrics.brier,
    ece: current.ece - baseline.metrics.ece,
  };
  const failures: string[] = [];
  if (deltas.accuracy < -thresholds.accuracy_drop) {
    failures.push(
      `accuracy regressed: ${(baseline.metrics.accuracy * 100).toFixed(1)}% -> ${(
        current.accuracy * 100
      ).toFixed(1)}% (allowed drop ${thresholds.accuracy_drop * 100}pt)`
    );
  }
  if (deltas.brier > thresholds.brier) {
    failures.push(
      `brier regressed: ${baseline.metrics.brier.toFixed(4)} -> ${current.brier.toFixed(
        4
      )} (allowed +${thresholds.brier})`
    );
  }
  if (deltas.ece > thresholds.ece) {
    failures.push(
      `ece regressed: ${baseline.metrics.ece.toFixed(4)} -> ${current.ece.toFixed(
        4
      )} (allowed +${thresholds.ece})`
    );
  }
  return { ok: failures.length === 0, failures, deltas };
}
