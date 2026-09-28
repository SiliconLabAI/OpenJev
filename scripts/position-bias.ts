/**
 * Position-bias robustness check.
 *
 * Runs the parallel readout over a JSONL dataset twice: once with the options
 * in the original order and once with the option order shuffled (deterministic
 * seed, reproducible across runs). Any argmax flip is reported.
 *
 * Input JSONL rows: { "state": string, "questions": {...} } — the same shape
 * as POST /api/evaluate, one per line.
 *
 *   npx tsx scripts/position-bias.ts data.jsonl [--base-url URL] [--model M] [--api-key K] [--readout logprob|micro]
 */
import { readFileSync } from "fs";
import { evaluate } from "../src/lib/evaluate";
import { shuffleChoiceCriteria } from "../src/lib/shuffle";
import { limitConcurrency } from "../src/lib/concurrency";
import type { EvaluateRequest } from "../src/lib/types";

interface Row {
  state: EvaluateRequest["state"];
  questions: EvaluateRequest["questions"];
}

function parseArgs(argv: string[]): Record<string, string> {
  const args: Record<string, string> = {};
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        args[key] = next;
        i++;
      } else {
        args[key] = "true";
      }
    }
  }
  return args;
}
async function main() {
  const inputPath = process.argv[2];
  if (!inputPath) {
    console.error(
      "Usage: npx tsx scripts/position-bias.ts <data.jsonl> [--base-url URL] [--model M] [--api-key K] [--readout logprob|micro]"
    );
    process.exit(1);
  }

  const rows: Row[] = readFileSync(inputPath, "utf8")
    .split(/\r?\n/)
    .filter((l) => l.trim() && !l.startsWith("//"))
    .map((l, i) => {
      try {
        return JSON.parse(l) as Row;
      } catch {
        throw new Error(`Invalid JSON on line ${i + 1} of ${inputPath}`);
      }
    });

  if (!rows.length) {
    console.error("No rows found in dataset");
    process.exit(1);
  }

  const args = parseArgs(process.argv);
  const model = args["model"] || process.env.OPENJEV_MODEL || "gpt-4o-mini";
  const readout = args["readout"] === "micro" ? "micro" : "logprob";
  const limit = Math.max(1, Number(process.env.OPENJEV_CONCURRENCY) || 6);

  // Bounded concurrency so large datasets don't trigger provider 429 storms.
  const results = await limitConcurrency(
    rows,
    limit,
    async (row, idx) => {
      const base: EvaluateRequest = {
        state: row.state,
        questions: row.questions,
        model,
        base_url: args["base-url"] || undefined,
        api_key: args["api-key"] || undefined,
        mode: "parallel",
        readout,
      };
      const original = await evaluate(base);
      const shuffled = await evaluate({
        ...base,
        questions: shuffleChoiceCriteria(row.questions),
      });

      const flips: string[] = [];
      for (const [name, answer] of Object.entries(original.answers)) {
        const s = shuffled.answers[name];
        if (!s) continue;
        if (answer.type === "choice" && s.type === "choice" && answer.choice !== s.choice) {
          flips.push(`${name}: ${answer.choice} -> ${s.choice}`);
        }
      }
      console.log(
        `[${idx + 1}/${rows.length}] flips=${flips.length}` +
          (flips.length ? ` :: ${flips.join("; ")}` : " (stable)")
      );
      return { row: idx + 1, flips };
    }
  );

  const totalFlips = results.reduce((acc, r) => acc + r.flips.length, 0);
  const rate = results.length ? (totalFlips / results.length) * 100 : 0;
  console.log("\n=== Position-bias summary ===");
  console.log(`rows: ${results.length}`);
  console.log(`rows with flipped answers: ${totalFlips}`);
  console.log(
    `flip rate: ${rate.toFixed(1)}% (share of rows whose choice changes when option order is shuffled)`
  );
  if (rate > 5) {
    console.log("high flip rate — inspect calibration temperature or prompt wording");
  }
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
