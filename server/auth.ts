import { createHash, timingSafeEqual } from "crypto";
import type { NextFunction, Request, Response } from "express";
import { logger } from "./logger";

function keyDigest(key: string): Buffer {
  return createHash("sha256").update(key).digest();
}

/**
 * Static bearer-token auth for /api/evaluate.
 *
 * Keys come from OPENJEV_API_KEYS (comma-separated). With no keys configured
 * the API is open — a sane default for local dev, logged as a warning once.
 */
export function createAuthMiddleware() {
  const digests = (process.env.OPENJEV_API_KEYS ?? "")
    .split(",")
    .map((k) => k.trim())
    .filter(Boolean)
    .map(keyDigest);

  if (digests.length === 0) {
    logger.warn(
      "OPENJEV_API_KEYS not set — /api/evaluate is unauthenticated (fine locally, NOT for production)"
    );
  }

  return (req: Request, res: Response, next: NextFunction) => {
    if (digests.length === 0) return next();
    const header = req.headers.authorization;
    const provided = header?.startsWith("Bearer ")
      ? header.slice(7).trim()
      : typeof req.headers["x-api-key"] === "string"
        ? (req.headers["x-api-key"] as string).trim()
        : "";
    if (provided) {
      const d = keyDigest(provided);
      for (const expected of digests) {
        if (d.length === expected.length && timingSafeEqual(d, expected)) {
          return next();
        }
      }
    }
    res
      .status(401)
      .json({ error: "Unauthorized: set Authorization: Bearer <key> (see OPENJEV_API_KEYS)" });
  };
}
