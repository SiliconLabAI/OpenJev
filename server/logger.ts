/**
 * Structured JSON-line logging.
 *
 * Every line is a single JSON object on stdout/stderr so logs are greppable
 * and shippable to any aggregator. Level is controlled by OPENJEV_LOG_LEVEL
 * (debug | info | warn | error; default info).
 */

export type LogLevel = "debug" | "info" | "warn" | "error";

const LEVELS: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };

function threshold(): number {
  const raw = process.env.OPENJEV_LOG_LEVEL as LogLevel | undefined;
  return (raw && LEVELS[raw]) || LEVELS.info;
}

function emit(
  base: Record<string, unknown>,
  level: LogLevel,
  msg: string,
  fields?: Record<string, unknown>
): void {
  if (LEVELS[level] < threshold()) return;
  const entry = { t: new Date().toISOString(), level, msg, ...base, ...fields };
  const line = JSON.stringify(entry, (_k, v) =>
    v instanceof Error ? { name: v.name, message: v.message } : v
  );
  if (level === "error") process.stderr.write(line + "\n");
  else process.stdout.write(line + "\n");
}

export interface Logger {
  debug(msg: string, fields?: Record<string, unknown>): void;
  info(msg: string, fields?: Record<string, unknown>): void;
  warn(msg: string, fields?: Record<string, unknown>): void;
  error(msg: string, fields?: Record<string, unknown>): void;
  child(fields: Record<string, unknown>): Logger;
}

export function createLogger(base: Record<string, unknown> = {}): Logger {
  return {
    debug: (msg, fields) => emit(base, "debug", msg, fields),
    info: (msg, fields) => emit(base, "info", msg, fields),
    warn: (msg, fields) => emit(base, "warn", msg, fields),
    error: (msg, fields) => emit(base, "error", msg, fields),
    child: (fields) => createLogger({ ...base, ...fields }),
  };
}

export const logger = createLogger({ svc: "openjev" });
