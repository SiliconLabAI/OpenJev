import "./loadEnv";
import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { evaluate } from "../src/lib/evaluate";
import type { EvaluateRequest } from "../src/lib/types";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isProd = process.env.NODE_ENV === "production";
const PORT = Number(process.env.PORT) || 3001;

function resolveEnvKey(): string | undefined {
  return (
    process.env.OPENAI_API_KEY ||
    process.env.OPENJEV_API_KEY ||
    process.env.CEREBRAS_API_KEY ||
    process.env.PUTER_API_KEY ||
    process.env.GROQ_API_KEY ||
    process.env.API_KEY ||
    undefined
  );
}

function resolveEnvBaseUrl(): string | undefined {
  return (
    process.env.OPENAI_BASE_URL ||
    process.env.OPENJEV_BASE_URL ||
    process.env.BASE_URL ||
    undefined
  );
}

function resolveEnvModel(): string | undefined {
  return process.env.OPENJEV_MODEL || process.env.MODEL || undefined;
}

async function main() {
  const app = express();
  app.use(cors());
  app.use(express.json({ limit: "2mb" }));

  // Debug: confirm .env loaded (never log the key itself)
  const envKeyPresent = Boolean(resolveEnvKey());
  console.log(
    `[openjev] .env keys loaded: api_key=${envKeyPresent ? "yes" : "no"}, ` +
      `base_url=${resolveEnvBaseUrl() ? "yes" : "no"}, ` +
      `model=${resolveEnvModel() ?? "(default)"}`
  );

  /**
   * POST /api/evaluate
   * OpenJev System One endpoint (Jev-compatible request shape).
   * Default mode=parallel scores each option independently then normalizes.
   *
   * API key resolution order:
   *   1. body.api_key (from UI)
   *   2. .env / process.env (OPENAI_API_KEY, CEREBRAS_API_KEY, …)
   */
  app.post("/api/evaluate", async (req, res) => {
    try {
      const body = req.body as EvaluateRequest;

      if (body.state === undefined || body.state === null) {
        res.status(400).json({ error: "Missing 'state'" });
        return;
      }
      if (!body.questions || typeof body.questions !== "object") {
        res.status(400).json({ error: "Missing 'questions' object" });
        return;
      }

      const api_key =
        (typeof body.api_key === "string" && body.api_key.trim()) ||
        resolveEnvKey();
      const base_url =
        (typeof body.base_url === "string" && body.base_url.trim()) ||
        resolveEnvBaseUrl();
      const model =
        (typeof body.model === "string" && body.model.trim()) ||
        resolveEnvModel() ||
        "gpt-4o-mini";

      if (!api_key) {
        res.status(401).json({
          error:
            "No API key. Set OPENAI_API_KEY (or CEREBRAS_API_KEY / PUTER_API_KEY / GROQ_API_KEY) in .env, or paste a key in the UI.",
        });
        return;
      }

      const result = await evaluate({
        state: body.state,
        questions: body.questions,
        model,
        base_url,
        api_key,
        temperature: body.temperature ?? 0,
        mode: body.mode ?? "parallel",
      });

      res.json(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[/api/evaluate]", message);
      res.status(500).json({
        error: message,
        model: (req.body as EvaluateRequest)?.model ?? "unknown",
        answers: {},
        usage: { input_tokens: null, output_tokens: null },
      });
    }
  });

  app.get("/api/health", (_req, res) => {
    res.json({
      ok: true,
      service: "openjev",
      env_key: Boolean(resolveEnvKey()),
    });
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

  app.listen(PORT, () => {
    console.log(`\n  OpenJev Playground`);
    console.log(`  → http://localhost:${PORT}`);
    console.log(`  → POST /api/evaluate  (parallel sampler by default)\n`);
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
