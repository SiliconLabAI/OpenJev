/**
 * Retry helper for LLM calls: exponential backoff with jitter, honors
 * Retry-After, and fails fast on non-retryable client errors (4xx).
 *
 * The OpenAI SDK client is configured with maxRetries: 0, so this module
 * owns all retry decisions.
 */

export interface RetryOptions {
  /** Total attempts including the first one. Default: OPENJEV_MAX_RETRIES or 3. */
  attempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  sleep: (ms: number) => Promise<void>;
  onRetry?: (info: { attempt: number; delayMs: number; error: unknown }) => void;
}

function toInt(raw: string | undefined, fallback: number): number {
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

export function defaultRetryOptions(): RetryOptions {
  return {
    attempts: toInt(process.env.OPENJEV_MAX_RETRIES, 3),
    baseDelayMs: 400,
    maxDelayMs: 8000,
    sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  };
}

/** 408/429/5xx and network-level errors (no status) are retryable; other 4xx fail fast. */
export function isRetryableError(err: unknown): boolean {
  const status = (err as { status?: number | null } | null)?.status;
  if (typeof status === "number") {
    return status === 408 || status === 429 || status >= 500;
  }
  // No status: timeouts / connection errors from the SDK.
  return true;
}

/** Extract a Retry-After hint (seconds or HTTP date) from an error's headers, if any. */
export function retryAfterMs(err: unknown, maxDelayMs: number): number | null {
  const headers = (err as { headers?: unknown } | null)?.headers;
  if (!headers) return null;
  let value: string | number | null | undefined;
  if (typeof (headers as Headers).get === "function") {
    value = (headers as Headers).get("retry-after");
  } else if (typeof headers === "object") {
    const rec = headers as Record<string, string | number | undefined>;
    value = rec["retry-after"] ?? rec["Retry-After"];
  }
  if (value === null || value === undefined) return null;
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.min(maxDelayMs, Math.max(0, value * 1000));
  }
  const asSeconds = Number(value);
  if (Number.isFinite(asSeconds)) return Math.min(maxDelayMs, Math.max(0, asSeconds * 1000));
  const asDate = Date.parse(String(value));
  if (!Number.isNaN(asDate)) {
    return Math.min(maxDelayMs, Math.max(0, asDate - Date.now()));
  }
  return null;
}

/**
 * Run `fn` with retries. Returns the result plus how many retries were used.
 * Throws the last error when attempts are exhausted or the error is non-retryable.
 */
export async function withRetry<T>(
  label: string,
  fn: () => Promise<T>,
  opts?: Partial<RetryOptions>
): Promise<{ result: T; retries: number }> {
  const o: RetryOptions = { ...defaultRetryOptions(), ...opts };
  let lastErr: unknown;
  for (let attempt = 1; attempt <= o.attempts; attempt++) {
    try {
      return { result: await fn(), retries: attempt - 1 };
    } catch (err) {
      lastErr = err;
      if (attempt === o.attempts || !isRetryableError(err)) break;
      const hinted = retryAfterMs(err, o.maxDelayMs);
      const backoff = Math.min(
        o.maxDelayMs,
        o.baseDelayMs * 2 ** (attempt - 1) * (0.5 + Math.random() / 2)
      );
      const delay = hinted ?? backoff;
      o.onRetry?.({ attempt, delayMs: Math.round(delay), error: err });
      console.warn(
        `[openjev] ${label} attempt ${attempt}/${o.attempts} failed (${
          err instanceof Error ? err.message : String(err)
        }) — retrying in ${Math.round(delay)}ms`
      );
      await o.sleep(delay);
    }
  }
  throw lastErr;
}
