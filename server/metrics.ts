/**
 * Lightweight in-process metrics.
 *
 * Counters plus a bounded latency sample for percentiles, exposed as JSON at
 * GET /api/metrics. Good enough for a single-process service; swap for
 * prom-client + a registry when this scales beyond one instance.
 */
import type { EvaluateResponse } from "../src/lib/types";

const MAX_LATENCY_SAMPLES = 1000;

interface State {
  startedAt: number;
  requestsTotal: number;
  badRequests: number;
  evaluateTotal: number;
  evaluateOk: number;
  evaluateError: number;
  fallbackUsedTotal: number;
  fallbackQuestionsTotal: number;
  failedQuestionsTotal: number;
  questionsTotal: number;
  retriesTotal: number;
  tokensInTotal: number;
  tokensOutTotal: number;
  latencySamples: number[];
}

const state: State = {
  startedAt: Date.now(),
  requestsTotal: 0,
  badRequests: 0,
  evaluateTotal: 0,
  evaluateOk: 0,
  evaluateError: 0,
  fallbackUsedTotal: 0,
  fallbackQuestionsTotal: 0,
  failedQuestionsTotal: 0,
  questionsTotal: 0,
  retriesTotal: 0,
  tokensInTotal: 0,
  tokensOutTotal: 0,
  latencySamples: [],
};

export function incRequest(): void {
  state.requestsTotal++;
}

export function incBadRequest(): void {
  state.badRequests++;
}

export function recordEvaluateResult(res: EvaluateResponse): void {
  state.evaluateTotal++;
  state.evaluateOk++;
  const meta = res.meta;
  if (meta) {
    state.latencySamples.push(meta.latency_ms);
    if (state.latencySamples.length > MAX_LATENCY_SAMPLES) {
      state.latencySamples.splice(0, state.latencySamples.length - MAX_LATENCY_SAMPLES);
    }
    if (meta.fallback_used) state.fallbackUsedTotal++;
    state.fallbackQuestionsTotal += meta.fallback_questions ?? 0;
    state.failedQuestionsTotal += meta.failed_questions?.length ?? 0;
    state.retriesTotal += meta.retries ?? 0;
    state.questionsTotal += Object.keys(res.answers).length;
  }
  state.tokensInTotal += res.usage.input_tokens ?? 0;
  state.tokensOutTotal += res.usage.output_tokens ?? 0;
}

export function recordEvaluateError(): void {
  state.evaluateTotal++;
  state.evaluateError++;
}

function percentile(sorted: number[], p: number): number | null {
  if (sorted.length === 0) return null;
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

export function snapshot(): Record<string, unknown> {
  const sorted = [...state.latencySamples].sort((a, b) => a - b);
  return {
    uptime_s: Math.round((Date.now() - state.startedAt) / 1000),
    requests_total: state.requestsTotal,
    bad_requests: state.badRequests,
    evaluate_total: state.evaluateTotal,
    evaluate_ok: state.evaluateOk,
    evaluate_error: state.evaluateError,
    questions_total: state.questionsTotal,
    failed_questions_total: state.failedQuestionsTotal,
    fallback_used_total: state.fallbackUsedTotal,
    fallback_questions_total: state.fallbackQuestionsTotal,
    retries_total: state.retriesTotal,
    tokens_in_total: state.tokensInTotal,
    tokens_out_total: state.tokensOutTotal,
    latency_ms: {
      count: sorted.length,
      p50: percentile(sorted, 50),
      p95: percentile(sorted, 95),
    },
  };
}
