import { test, describe } from "node:test";
import assert from "node:assert/strict";

import {
  parseLetterLogprobs,
  parseYesNoLogprobs,
  buildOptionLetterPrompt,
} from "./logprob";
import {
  softmaxFromLogits,
  temperatureScaleProbs,
  resolveCalibrationT,
  brierScore,
  expectedCalibrationError,
  fitTemperature,
} from "./calibration";
import { withRetry, isRetryableError, retryAfterMs } from "./retry";
import { withFallback } from "./fallback";
import { shuffleDeterministic, shuffleChoiceCriteria } from "./shuffle";

// ── logprob parsing ────────────────────────────────────────────────────────

test("parseLetterLogprobs maps letters to logits", () => {
  const top = [
    { token: " B", logprob: -0.1 },
    { token: "A", logprob: -2.5 },
    { token: "C", logprob: -4.0 },
    { token: "D", logprob: -6.0 },
  ];
  const { logits, matched } = parseLetterLogprobs(top, 4);
  assert.equal(matched, 4);
  assert.deepEqual(logits, [-2.5, -0.1, -4.0, -6.0]);
});

test("parseLetterLogprobs floors missing letters below the minimum seen", () => {
  const top = [
    { token: "A", logprob: -0.2 },
    { token: "B", logprob: -1.0 },
  ];
  const { logits, matched } = parseLetterLogprobs(top, 3);
  assert.equal(matched, 2);
  // Floor = min seen logit (-1.0) minus floorGap (5) = -6 exactly.
  assert.equal(logits[2], -6);
});

test("parseLetterLogprobs tolerates punctuation and whitespace variants", () => {
  const top = [
    { token: "A)", logprob: -0.3 },
    { token: " B.", logprob: -0.9 },
    { token: "\tC", logprob: -2.0 },
    { token: "D", logprob: -3.0 },
  ];
  const { logits, matched } = parseLetterLogprobs(top, 4);
  assert.equal(matched, 4);
  assert.deepEqual(logits, [-0.3, -0.9, -2.0, -3.0]);
});

test("parseLetterLogprobs ignores non-letter tokens", () => {
  const top = [
    { token: "1", logprob: -0.1 },
    { token: " Yes", logprob: -0.2 },
    { token: "A", logprob: -0.5 },
  ];
  const { logits, matched } = parseLetterLogprobs(top, 2);
  assert.equal(matched, 1);
  assert.ok(logits[0] === -0.5);
  assert.ok(logits[1] < -5); // floored
});

test("parseYesNoLogprobs finds yes/no tokens", () => {
  const top = [
    { token: " yes", logprob: -0.4 },
    { token: "No", logprob: -1.2 },
  ];
  const { yesLogit, noLogit } = parseYesNoLogprobs(top);
  assert.equal(yesLogit, -0.4);
  assert.equal(noLogit, -1.2);
});

test("parseYesNoLogprobs returns nulls when absent", () => {
  const top = [{ token: " maybe", logprob: -0.4 }];
  const { yesLogit, noLogit } = parseYesNoLogprobs(top);
  assert.equal(yesLogit, null);
  assert.equal(noLogit, null);
});

test("buildOptionLetterPrompt lists letters, descriptions and range", () => {
  const p = buildOptionLetterPrompt("S", "Q", [
    { key: "billing", description: "charges" },
    { key: "technical", description: "bugs" },
  ]);
  assert.ok(p.user.includes("A. billing — charges"));
  assert.ok(p.user.includes("B. technical — bugs"));
  assert.ok(p.user.includes("(A–B)"));
});

// ── calibration ────────────────────────────────────────────────────────────

test("temperatureScaleProbs flattens when T>1 and sharpens when T<1", () => {
  const probs = [0.9, 0.1];
  const flat = temperatureScaleProbs(probs, 2);
  const sharp = temperatureScaleProbs(probs, 0.5);
  assert.ok(flat[0] < 0.9 && flat[0] > 0.5);
  assert.ok(sharp[0] > 0.9);
  assert.ok(Math.abs(flat[0] + flat[1] - 1) < 1e-9);
  assert.ok(Math.abs(sharp[0] + sharp[1] - 1) < 1e-9);
});

test("temperatureScaleProbs is identity at T=1", () => {
  const probs = [0.7, 0.2, 0.1];
  const same = temperatureScaleProbs(probs, 1);
  assert.deepEqual(same, probs);
});

test("resolveCalibrationT defaults to 1 and clamps to [0.1, 10]", () => {
  assert.equal(resolveCalibrationT(undefined, ""), 1);
  assert.equal(resolveCalibrationT(undefined, "2"), 2);
  // An explicit 0 is invalid and falls back to the default, not the env value.
  assert.equal(resolveCalibrationT(0, "5"), 1);
  assert.equal(resolveCalibrationT(20, ""), 10);
  assert.equal(resolveCalibrationT(-3, ""), 1);
  assert.equal(resolveCalibrationT(0.05, ""), 0.1);
});

test("brierScore is 0 for perfect and 2 for worst case", () => {
  assert.equal(brierScore([1, 0], 0), 0);
  assert.equal(brierScore([0, 1], 0), 2);
  const b = brierScore([0.8, 0.2], 0);
  assert.ok(b > 0 && b < 0.1);
});

test("expectedCalibrationError is 0 when confidence matches accuracy", () => {
  const samples = [
    { probs: [0.75, 0.25], correctIdx: 0 },
    { probs: [0.75, 0.25], correctIdx: 1 },
    { probs: [0.75, 0.25], correctIdx: 0 },
    { probs: [0.75, 0.25], correctIdx: 0 },
  ];
  // 3/4 = 75% accuracy at 0.75 mean confidence within one bucket → ECE ~ 0.
  assert.ok(expectedCalibrationError(samples) < 1e-9);
});

test("fitTemperature recovers a known temperature", () => {
  // Model is overconfident: it outputs softmax([4,0]) ≈ [0.982, 0.018] but is
  // only right at the rate of the T=2-scaled distribution (≈ 0.881). The NLL
  // optimum then sits at T=2 exactly. All-same labels would push the MLE to
  // the grid edge instead, so labels must be mixed in the true proportions.
  const sharp = softmaxFromLogits([4, 0], 1);
  const trueRate = temperatureScaleProbs(sharp, 2);
  const n = 1000;
  const ones = Math.round(trueRate[1] * n); // 119 of 1000
  const samples = Array.from({ length: n }, (_, i) => ({
    probs: [...sharp],
    correctIdx: i < ones ? 1 : 0,
  }));
  const fit = fitTemperature(samples);
  assert.ok(
    fit.temperature >= 1.95 && fit.temperature <= 2.05,
    `got ${fit.temperature}`
  );
});

// ── retry ──────────────────────────────────────────────────────────────────

test("withRetry retries retryable errors then succeeds", async () => {
  let calls = 0;
  const { result, retries } = await withRetry(
    "t",
    async () => {
      calls++;
      if (calls < 3) {
        const e = new Error("boom");
        (e as { status?: number }).status = 429;
        throw e;
      }
      return "ok";
    },
    { attempts: 3, baseDelayMs: 1, maxDelayMs: 2, sleep: async () => {} }
  );
  assert.equal(result, "ok");
  assert.equal(retries, 2);
  assert.equal(calls, 3);
});

test("withRetry fails fast on non-retryable 4xx", async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(
      "t",
      async () => {
        calls++;
        const e = new Error("bad request");
        (e as { status?: number }).status = 400;
        throw e;
      },
      { attempts: 3, baseDelayMs: 1, maxDelayMs: 2, sleep: async () => {} }
    )
  );
  assert.equal(calls, 1);
});

test("withRetry throws the last error when attempts are exhausted", async () => {
  let calls = 0;
  await assert.rejects(
    withRetry(
      "t",
      async () => {
        calls++;
        const e = new Error("still down");
        (e as { status?: number }).status = 500;
        throw e;
      },
      { attempts: 2, baseDelayMs: 1, maxDelayMs: 2, sleep: async () => {} }
    )
  );
  assert.equal(calls, 2);
});

test("isRetryableError: 408/429/5xx and network errors are retryable, other 4xx are not", () => {
  const mk = (status?: number) => {
    const e = new Error("x");
    (e as { status?: number | null }).status = status ?? null;
    return e;
  };
  assert.equal(isRetryableError(mk(408)), true);
  assert.equal(isRetryableError(mk(429)), true);
  assert.equal(isRetryableError(mk(500)), true);
  assert.equal(isRetryableError(mk(400)), false);
  assert.equal(isRetryableError(mk(401)), false);
  assert.equal(isRetryableError(new Error("network")), true);
});

test("retryAfterMs parses seconds and clamps to maxDelay", () => {
  assert.equal(retryAfterMs({ headers: { "retry-after": "2" } }, 8000), 2000);
  assert.equal(retryAfterMs({ headers: { "retry-after": "999" } }, 8000), 8000);
  assert.equal(retryAfterMs({}, 8000), null);
  assert.equal(retryAfterMs({ headers: { "retry-after": "invalid" } }, 8000), null);
});

// ── fallback ───────────────────────────────────────────────────────────────

test("withFallback returns first success and null used", async () => {
  const { result, used } = await withFallback<string>([
    { label: "a", run: async () => "from-a" },
    { label: "b", run: async () => "from-b" },
  ]);
  assert.equal(result, "from-a");
  assert.equal(used, null);
});

test("withFallback falls back to the next step and records it", async () => {
  const logs: string[] = [];
  const { result, used } = await withFallback<string>(
    [
      { label: "a", run: async () => { throw new Error("a failed"); } },
      { label: "b", run: async () => "from-b" },
    ],
    { log: (m) => logs.push(m) }
  );
  assert.equal(result, "from-b");
  assert.equal(used, "b");
  assert.ok(logs.some((l) => l.includes("a failed")));
});

test("withFallback throws the last error when all steps fail", async () => {
  await assert.rejects(
    withFallback<string>([
      { label: "a", run: async () => { throw new Error("a failed"); } },
      { label: "b", run: async () => { throw new Error("b failed"); } },
    ])
  );
});

// ── shuffle ────────────────────────────────────────────────────────────────

test("shuffleDeterministic is reproducible and preserves elements", () => {
  const items = ["a", "b", "c", "d", "e"];
  const one = shuffleDeterministic(items, 42);
  const two = shuffleDeterministic(items, 42);
  assert.deepEqual(one, two);
  assert.deepEqual([...one].sort(), items);
});

describe("shuffleChoiceCriteria", () => {
  test("shuffles choice criteria but leaves other questions", () => {
    const questions = {
      dept: {
        type: "choice" as const,
        instructions: "Which team?",
        criteria: { billing: "charges", technical: "bugs", account: "access" },
      },
      angry: { type: "noul" as const, instructions: "Angry?" },
    };
    const shuffled = shuffleChoiceCriteria(questions, 7);
    assert.equal(shuffled.angry, questions.angry);
    if (shuffled.dept.type !== "choice") {
      throw new Error("dept should remain a choice question");
    }
    // Same entries, possibly reordered; descriptions stay attached to their keys.
    assert.deepEqual(
      Object.keys(shuffled.dept.criteria).sort(),
      Object.keys(questions.dept.criteria).sort()
    );
    assert.deepEqual(shuffled.dept.criteria, questions.dept.criteria);
  });
});
