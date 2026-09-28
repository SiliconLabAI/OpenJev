import "./loadEnv";
import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { randomUUID } from "crypto";
import { evaluate } from "../src/lib/evaluate";
import type { EvaluateRequest } from "../src/lib/types";
import { logger } from "./logger";
import { createAuthMiddleware } from "./auth";
import { createRateLimiter } from "./rateLimit";
import {
  parseEvaluateRequest,
  assertPublicUrl,
  ValidationError,
} from "./validation";
import {
  incBadRequest,
  incRequest,
  recordEvaluateError,
  recordEvaluateResult,
  snapshot,
} from "./metrics";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProd = process.env.NODE_ENV === "production";
const PORT = Number(process.env.PORT) || 3001;

async function main() {
  const app = express();
  app.use(cors());
  app.use(express.json({ limit: "2mb" }));

  logger.info("env", {
    has_openai_key: Boolean(
      process.env.OPENAI_API_KEY ||
        process.env.CEREBRAS_API_KEY ||
        process.env.GROQ_API_KEY
    ),
    decider_url: process.env.DECIDER_BASE_URL || "http://localhost:8000",
    auth_enabled: Boolean((process.env.OPENJEV_API_KEYS ?? "").trim()),
    is_prod: isProd,
  });

  // Health/metrics stay unauthenticated (orchestrator probes); the evaluate
  // endpoint is rate-limited, then auth-gated.
  app.post("/api/evaluate", createRateLimiter(), createAuthMiddleware());

  app.post("/api/evaluate", async (req, res) => {
    const t0 = Date.now();
    incRequest();
    // Abort in-flight LLM calls when the client disconnects. Must listen on
    // the response: req's own 'close' fires as soon as the body is consumed.
    const abort = new AbortController();
    res.on("close", () => {
      if (!res.writableEnded) abort.abort();
    });

    try {
      const body = parseEvaluateRequest(req.body);
      if (body.base_url) assertPublicUrl(body.base_url, "base_url");
      if (body.decider_url) assertPublicUrl(body.decider_url, "decider_url");

      const log = logger.child({
        req_id:
          (req.headers["x-request-id"] as string | undefined) || randomUUID(),
      });
      log.debug("evaluate request", {
        mode: body.mode ?? "parallel",
        model: body.model,
        questions: Object.keys(body.questions).length,
      });

      const result = await evaluate({
        state: body.state,
        questions: body.questions,
        model: body.model,
        base_url: body.base_url,
        api_key: body.api_key,
        temperature: body.temperature ?? 0,
        mode: body.mode ?? "parallel",
        decider_url: body.decider_url,
        calibration_temperature: body.calibration_temperature,
        readout: body.readout,
        signal: abort.signal,
      });

      recordEvaluateResult(result);
      log.info("evaluate ok", {
        mode: result.meta?.mode,
        readout: result.meta?.readout,
        latency_ms: result.meta?.latency_ms,
        retries: result.meta?.retries ?? 0,
        fallback_used: result.meta?.fallback_used ?? null,
        failed_questions: result.meta?.failed_questions?.length ?? 0,
        wall_ms: Date.now() - t0,
      });
      res.json(result);
    } catch (err: unknown) {
      if (err instanceof ValidationError) {
        incBadRequest();
        res.status(400).json({ error: err.message, issues: err.issues });
        return;
      }
      recordEvaluateError();
      const message = err instanceof Error ? err.message : String(err);
      logger.error("evaluate failed", { message, wall_ms: Date.now() - t0 });
      res.status(500).json({
        error: message,
        model: (req.body as EvaluateRequest | undefined)?.model ?? "unknown",
        answers: {},
        usage: { input_tokens: null, output_tokens: null },
      });
    }
  });

  app.get("/api/health", (_req, res) => {
    res.json({
      ok: true,
      service: "openjev",
      backends: ["parallel", "oneshot", "decider"],
      decider_url: process.env.DECIDER_BASE_URL || "http://localhost:8000",
    });
  });

  app.get("/api/metrics", (_req, res) => {
    res.json(snapshot());
  });

  if (isProd) {
    app.use(express.static(path.join(__dirname, "../dist")));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(__dirname, "../dist/index.html"));
    });
  } else {
    const vite = await createViteServer({
      root: path.join(__dirname, ".."),
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  const server = app.listen(PORT, () => {
    logger.info("listening", { port: PORT, is_prod: isProd });
  });

  // Graceful shutdown: stop accepting, drain in-flight requests, force-exit.
  let draining = false;
  const shutdown = (sig: string) => {
    if (draining) return;
    draining = true;
    logger.info("shutting down", { signal: sig });
    server.close(() => process.exit(0));
    setTimeout(() => {
      logger.warn("shutdown timed out; forcing exit");
      process.exit(0);
    }, 10_000).unref();
  };
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}

main().catch((e) => {
  logger.error("fatal", { err: e instanceof Error ? e.stack : String(e) });
  process.exit(1);
});
