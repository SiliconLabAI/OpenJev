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

async function main() {
  const app = express();
  app.use(cors());
  app.use(express.json({ limit: "2mb" }));

  // ── REST API: POST /api/evaluate ──────────────────────────────────────
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

      // Prefer client-supplied key/url, then env
      const result = await evaluate({
        state: body.state,
        questions: body.questions,
        model: body.model,
        base_url: body.base_url,
        api_key: body.api_key,
        temperature: body.temperature ?? 0,
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

  // Health
  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, service: "openjev" });
  });

  if (isProd) {
    app.use(express.static(path.join(__dirname, "../dist")));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(__dirname, "../dist/index.html"));
    });
  } else {
    // Dev: Vite middleware mode so one process serves UI + API
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
    console.log(`  → POST /api/evaluate\n`);
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
