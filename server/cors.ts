import type { CorsOptions } from "cors";

export function parseAllowedOrigins(raw: string | undefined): string[] {
  if (!raw) return [];
  return raw
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

export function createCorsOptions(raw: string | undefined): CorsOptions {
  const allowed = new Set(parseAllowedOrigins(raw));
  return {
    origin(requestOrigin, callback) {
      if (!requestOrigin) {
        callback(null, true);
        return;
      }
      callback(null, allowed.has(requestOrigin));
    },
  };
}
