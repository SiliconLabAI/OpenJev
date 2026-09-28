/**
 * Golden-set evaluation CLI.
 *
 * Runs a labeled JSONL dataset through evaluate() and reports accuracy, Brier
 * and ECE. With --create it writes a baseline; without it, compares against
 * the stored baseline and exits non-zero on regression (CI-friendly).
 *
 *   npm run eval:golden                          # compare against baseline
 *   npm run eval:golden -- --create              # (re)create the baseline
 *   npx tsx scripts/eval-golden.ts data.jsonl --model gpt-4o-mini --base-url http://localhost:1234/v1
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import path from "path";
import { evaluate } from "../src/lib/evaluate";
import { limitConcurrency } from "../src/lib/concurrency";
import {
  parseGoldenRow,
  scoreResponse,
  aggregateMetrics,
  compareAgainstBaseline,
  type GoldenRow,
  type GoldenBaseline,
  type GoldenMetrics,
} from "../src/lib/golden";
import type { EvaluateRequest } from "../src/lib/types";

function parseArgs(argv: string[]): Record<string, string | boolean | undefined> {
  const args: Record<string, string | boolean | undefined> = {};
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith("--")) {
        args[key] = next;
        i++;
      } else {
        args[key] = true;
      }
    }
  }
  return args;
}

function str(args: Record<string, string | boolean | undefined>, key: string): string | undefined {
  const v = args[key];
  return typeof v === "string" && v.trim() !== "" ? v : undefined;
}

function num(args: Record<string, string | boolean | undefined>, key: string): number | undefined {
  const v = str(args, key);
  if (v === undefined) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

async function main() {
  const args = parseArgs(process.argv);
  const inputPath = str(args, "dataset") ?? process.argv[2] ?? "";
  if (!inputPath || inputPath.startsWith("--")) {
    console.error(
      "Usage: npx tsx scripts/eval-golden.ts <dataset.jsonl> [--create] [--baseline PATH]\n" +
        "       [--model M] [--base-url URL] [--api-key K] [--readout logprob|micro] [--calibration-t T]"
    );
    process.exit(1);
  }

  const baselinePath = str(args, "baseline") ?? "eval/golden/baseline.json";
  const create = args.create === true;

  const rows: GoldenRow[] = readFileSync(inputPath, "utf8")
    .split(/\r?\n/)
    .map((l, i) => ({ line: i + 1, text: l }))
    .filter(({ text }) => text.trim() && !text.trim().startsWith("//"))
    .map(({ line, text }) => parseGoldenRow(JSON.parse(text), line, inputPath));

  if (!rows.length) {
    console.error("No rows found in dataset");
    process.exit(1);
  }

  const model = str(args, "model") ?? process.env.OPENJEV_MODEL ?? "gpt-4o-mini";
  const readout = args.readout === "micro" ? ("micro" as const) : ("logprob" as const);
  const calibrationT = num(args, "calibration-t");
  const limit = Math.max(1, Number(process.env.OPENJEV_CONCURRENCY) || 4);

  console.log(`eval-golden: ${rows.length} rows, model=${model}, readout=${readout}`);

  let rowErrors = 0;
  const results = await limitConcurrency(rows, limit, async (row, idx) => {
    const request: EvaluateRequest = {
      state: row.state,
      questions: row.questions,
      model,
      base_url: str(args, "base-url"),
      api_key: str(args, "api-key"),
      mode: "parallel",
      readout,
      ...(calibrationT !== undefined ? { calibration_temperature: calibrationT } : {}),
    };
    try {
      const response = await evaluate(request);
      const { samples, scored, skipped } = scoreResponse(row, response);
      console.log(
        `[${idx + 1}/${rows.length}]${row.name ? ` ${row.name}:` : ""} scored=${scored} skipped=${skipped}`
      );
      return { samples, scored, skipped };
    } catch (err) {
      rowErrors++;
      console.error(
        `[${idx + 1}/${rows.length}]${row.name ? ` ${row.name}:` : ""} row failed: ${
          err instanceof Error ? err.message : String(err)
        }`
      );
      const skipped = Object.keys(row.expectations).length;
      return { samples: [], scored: 0, skipped };
    }
  });

  const samples = results.flatMap((r) => r.samples);
  const scored = results.reduce((a, r) => a + r.scored, 0);
  const skipped = results.reduce((a, r) => a + r.skipped, 0);
  const totalQuestions = scored + skipped;

  if (scored === 0) {
    console.error(
      `\nno questions could be scored (${rowErrors} row(s) errored) — check provider health, api key, and expectations`
    );
    process.exit(1);
  }

  const metrics: GoldenMetrics = {
    ...aggregateMetrics(samples),
    questions_total: totalQuestions,
    questions_skipped: skipped,
  };

  console.log("\n=== Golden-set metrics ===");
  console.log(
    `rows: ${rows.length}  questions: ${totalQuestions} (scored ${scored}, skipped ${skipped})`
  );
  console.log(`accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`);
  console.log(`brier:    ${metrics.brier.toFixed(4)}`);
  console.log(`ece:      ${metrics.ece.toFixed(4)}`);

  if (create) {
    const baseline: GoldenBaseline = {
      model,
      created_at: new Date().toISOString(),
      dataset_path: inputPath,
      metrics: {
        questions_scored: scored,
        accuracy: metrics.accuracy,
        brier: metrics.brier,
        ece: metrics.ece,
      },
    };
    mkdirSync(path.dirname(baselinePath), { recursive: true });
    writeFileSync(baselinePath, JSON.stringify(baseline, null, 2) + "\n");
    console.log(`\nbaseline written to ${baselinePath}`);
    process.exit(0);
  }

  if (!existsSync(baselinePath)) {
    console.error(
      `\nno baseline at ${baselinePath} — create one first with: npm run eval:golden -- --create`
    );
    process.exit(1);
  }

  const baseline = JSON.parse(readFileSync(baselinePath, "utf8")) as GoldenBaseline;
  const check = compareAgainstBaseline(metrics, baseline);

  console.log("\n=== Regression check vs baseline ===");
  console.log(
    `accuracy: ${(baseline.metrics.accuracy * 100).toFixed(1)}% -> ${(metrics.accuracy * 100).toFixed(1)}% (delta ${(check.deltas.accuracy * 100).toFixed(1)}pt)`
  );
  console.log(
    `brier:    ${baseline.metrics.brier.toFixed(4)} -> ${metrics.brier.toFixed(4)} (delta ${check.deltas.brier >= 0 ? "+" : ""}${check.deltas.brier.toFixed(4)})`
  );
  console.log(
    `ece:      ${baseline.metrics.ece.toFixed(4)} -> ${metrics.ece.toFixed(4)} (delta ${check.deltas.ece >= 0 ? "+" : ""}${check.deltas.ece.toFixed(4)})`
  );

  if (check.ok) {
    console.log("\nPASS: no regression against baseline");
    process.exit(0);
  }
  console.error("\nFAIL: regression detected:");
  for (const f of check.failures) console.error(`  - ${f}`);
  process.exit(1);
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
