import { test, describe } from "node:test";
import assert from "node:assert/strict";

import {
  parseGoldenRow,
  toFitSample,
  aggregateMetrics,
  scoreResponse,
  compareAgainstBaseline,
  REGRESSION_THRESHOLDS,
  type GoldenBaseline,
} from "./golden";
import type { Answer } from "./types";

describe("parseGoldenRow", () => {
  test("accepts a valid row", () => {
    const row = parseGoldenRow(
      {
        state: "s",
        questions: { q: { type: "noul", instructions: "x" } },
        expectations: { q: { noul: 1 } },
      },
      3,
      "data.jsonl"
    );
    assert.equal(row.state, "s");
  });

  test("reports file:line on missing fields", () => {
    assert.throws(
      () => parseGoldenRow({ questions: {}, expectations: {} }, 7, "data.jsonl"),
      /data\.jsonl:7.*state/
    );
    assert.throws(
      () => parseGoldenRow({ state: "s", questions: {} }, 7, "data.jsonl"),
      /data\.jsonl:7.*expectations/
    );
  });
});

describe("toFitSample", () => {
  test("choice: expectation key maps to the correct index", () => {
    const answer: Answer = {
      type: "choice",
      choice: "b",
      confidence: 0.7,
      probabilities: { a: 0.7, b: 0.3 },
    };
    const s = toFitSample(answer, { choice: "b" });
    assert.ok(s);
    assert.deepEqual(s.probs, [0.7, 0.3]);
    assert.equal(s.correctIdx, 1);
  });

  test("choice: unknown option key is not scorable", () => {
    const answer: Answer = {
      type: "choice",
      choice: "a",
      confidence: 1,
      probabilities: { a: 1, b: 0 },
    };
    assert.equal(toFitSample(answer, { choice: "zzz" }), null);
  });

  test("choice: missing probabilities is not scorable", () => {
    const answer: Answer = { type: "choice", choice: "a", confidence: 1, probabilities: {} };
    assert.equal(toFitSample(answer, { choice: "a" }), null);
  });

  test("score: expectation level maps to level index", () => {
    const answer: Answer = {
      type: "score",
      score: 1.2,
      confidence: 0.5,
      legend: { "0": "low", "1": "mid", "2": "high" },
      probabilities: { "0": 0.2, "1": 0.5, "2": 0.3 },
    };
    const s = toFitSample(answer, { score: 2 });
    assert.ok(s);
    assert.deepEqual(s.probs, [0.2, 0.5, 0.3]);
    assert.equal(s.correctIdx, 2);
  });

  test("score: out-of-range level is not scorable", () => {
    const answer: Answer = {
      type: "score",
      score: 0,
      confidence: 0.5,
      legend: {},
      probabilities: { "0": 1, "1": 0 },
    };
    assert.equal(toFitSample(answer, { score: 5 }), null);
  });

  test("noul: threshold at 0.5 splits yes/no classes", () => {
    const yes: Answer = { type: "noul", noul: 0.8 };
    const sYes = toFitSample(yes, { noul: 0.9 });
    assert.ok(sYes);
    // 1 - 0.8 is not exactly 0.2 in float math; compare with tolerance.
    assert.ok(Math.abs(sYes.probs[0] - 0.8) < 1e-9);
    assert.ok(Math.abs(sYes.probs[1] - 0.2) < 1e-9);
    assert.equal(sYes.correctIdx, 0); // expectation >= 0.5 -> "yes" is correct

    const no: Answer = { type: "noul", noul: 0.8 };
    const sNo = toFitSample(no, { noul: 0.1 });
    assert.ok(sNo);
    assert.equal(sNo.correctIdx, 1);
  });
});

describe("aggregateMetrics", () => {
  test("perfect predictions score 1.0 accuracy and 0 brier", () => {
    const m = aggregateMetrics([
      { probs: [1, 0], correctIdx: 0 },
      { probs: [0, 1], correctIdx: 1 },
    ]);
    assert.equal(m.accuracy, 1);
    assert.equal(m.brier, 0);
    assert.equal(m.ece, 0);
  });

  test("empty input yields zeros without dividing by NaN", () => {
    const m = aggregateMetrics([]);
    assert.equal(m.accuracy, 0);
    assert.equal(m.brier, 0);
    assert.equal(m.ece, 0);
  });

  test("known mixture produces exact accuracy and brier", () => {
    const m = aggregateMetrics([
      { probs: [0.8, 0.2], correctIdx: 0 },
      { probs: [0.3, 0.7], correctIdx: 1 },
      { probs: [0.6, 0.4], correctIdx: 1 }, // wrong argmax
    ]);
    assert.equal(m.accuracy, 2 / 3);
    // Brier = sum_i (p_i - y_i)^2. For the correct class y=1 -> (1-p)^2;
    // for the wrong class y=0 -> p^2. Each sample here contributes the same
    // value twice (complement pairs): 0.2^2, 0.3^2, 0.6^2 respectively.
    const expectedBrier =
      (2 * 0.2 * 0.2 + 2 * 0.3 * 0.3 + 2 * 0.6 * 0.6) / 3;
    assert.ok(Math.abs(m.brier - expectedBrier) < 1e-9);
  });
});

describe("scoreResponse", () => {
  test("skips missing answers and counts scored questions", () => {
    const row = parseGoldenRow(
      {
        state: "s",
        questions: {
          a: { type: "noul" },
          b: { type: "noul" },
        },
        expectations: {
          a: { noul: 1 },
          b: { noul: 0 },
          ghost: { noul: 1 },
        },
      },
      1,
      "t.jsonl"
    );
    const response = { answers: { a: { type: "noul", noul: 0.9 } as Answer } };
    const { scored, skipped } = scoreResponse(row, response);
    assert.equal(scored, 1);
    assert.equal(skipped, 2); // missing answer for b, and ghost never answered
  });
});

describe("compareAgainstBaseline", () => {
  const base: GoldenBaseline = {
    model: "m",
    metrics: { questions_scored: 10, accuracy: 0.9, brier: 0.2, ece: 0.05, questions_total: 10 },
  };

  test("identical metrics pass", () => {
    const r = compareAgainstBaseline({ accuracy: 0.9, brier: 0.2, ece: 0.05 }, base);
    assert.equal(r.ok, true);
    assert.deepEqual(r.failures, []);
  });

  test("accuracy drop beyond threshold fails", () => {
    const r = compareAgainstBaseline({ accuracy: 0.85, brier: 0.2, ece: 0.05 }, base);
    assert.equal(r.ok, false);
    assert.ok(r.failures[0].includes("accuracy"));
  });

  test("brier increase beyond threshold fails", () => {
    const r = compareAgainstBaseline({ accuracy: 0.9, brier: 0.23, ece: 0.05 }, base);
    assert.equal(r.ok, false);
    assert.ok(r.failures[0].includes("brier"));
  });

  test("ece increase beyond threshold fails", () => {
    const r = compareAgainstBaseline({ accuracy: 0.9, brier: 0.2, ece: 0.08 }, base);
    assert.equal(r.ok, false);
    assert.ok(r.failures[0].includes("ece"));
  });

  test("small changes within thresholds pass", () => {
    const r = compareAgainstBaseline(
      {
        accuracy: base.metrics.accuracy - REGRESSION_THRESHOLDS.accuracy_drop / 2,
        brier: base.metrics.brier + REGRESSION_THRESHOLDS.brier / 2,
        ece: base.metrics.ece + REGRESSION_THRESHOLDS.ece / 2,
      },
      base
    );
    assert.equal(r.ok, true);
  });
});
