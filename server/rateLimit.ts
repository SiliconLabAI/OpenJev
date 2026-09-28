import type { NextFunction, Request, Response } from "express";

interface RateLimitConfig {
  windowMs: number;
  max: number;
}

function readConfig(): RateLimitConfig | null {
  const raw = (process.env.OPENJEV_RATE_LIMIT ?? "").trim().toLowerCase();
  if (raw === "off" || raw === "0") return null;
  const max = Number(raw) > 0 ? Math.floor(Number(raw)) : 60;
  const windowRaw = Number(process.env.OPENJEV_RATE_WINDOW_MS);
  const windowMs = Number.isFinite(windowRaw) && windowRaw > 0 ? Math.floor(windowRaw) : 60_000;
  return { windowMs, max };
}

/**
 * Fixed-window, in-memory rate limiter keyed by client IP (per process).
 * OPENJEV_RATE_LIMIT = requests per OPENJEV_RATE_WINDOW_MS (default 60/min);
 * set OPENJEV_RATE_LIMIT=off to disable.
 */
export function createRateLimiter() {
  const cfg = readConfig();
  if (!cfg) {
    return (_req: Request, _res: Response, next: NextFunction) => next();
  }

  const hits = new Map<string, number[]>();
  const sweeper = setInterval(() => {
    const cutoff = Date.now() - cfg.windowMs;
    for (const [key, times] of hits) {
      const kept = times.filter((t) => t > cutoff);
      if (kept.length === 0) hits.delete(key);
      else hits.set(key, kept);
    }
  }, cfg.windowMs);
  sweeper.unref();

  return (req: Request, res: Response, next: NextFunction) => {
    const key = req.ip ?? "unknown";
    const now = Date.now();
    const cutoff = now - cfg.windowMs;
    const times = (hits.get(key) ?? []).filter((t) => t > cutoff);
    times.push(now);
    hits.set(key, times);
    if (times.length > cfg.max) {
      res.setHeader("retry-after", String(Math.ceil(cfg.windowMs / 1000)));
      res.status(429).json({ error: "Rate limit exceeded; slow down." });
      return;
    }
    next();
  };
}
