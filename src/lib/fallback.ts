/**
 * Fallback chain: try ordered steps, first success wins, later steps are
 * recorded as "fallback used" so callers can surface it in response meta.
 */

export interface FallbackStep<T> {
  label: string;
  run: () => Promise<T>;
}

export interface FallbackOptions {
  log?: (message: string) => void;
}

export async function withFallback<T>(
  steps: FallbackStep<T>[],
  opts?: FallbackOptions
): Promise<{ result: T; used: string | null }> {
  const log = opts?.log ?? ((m: string) => console.warn(m));
  let lastErr: unknown;
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    try {
      const result = await step.run();
      return { result, used: i === 0 ? null : step.label };
    } catch (err) {
      lastErr = err;
      const msg = err instanceof Error ? err.message : String(err);
      const next = i + 1 < steps.length ? ` — falling back to ${steps[i + 1].label}` : "";
      log(`[openjev] ${step.label} failed: ${msg}${next}`);
    }
  }
  throw lastErr;
}
