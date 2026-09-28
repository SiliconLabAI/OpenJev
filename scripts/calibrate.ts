/**
 * Calibration CLI.
 *
 * Fits a softmax temperature on held-out decision data (minimizes mean NLL)
 * and reports Brier score and ECE before and after scaling. Set the fitted
 * value as OPENJEV_CALIBRATION_T in the environment (or pass
 * calibration_temperature per request) to apply it at serve time.
 *
 * Input JSONL rows: { "probs": number[], "correctIdx": number } — e.g. the
 * choice distributions exported from a held-out eval run.
 *
 *   npx tsx scripts/calibrate.ts heldout.jsonl
 */
import { readFileSync } from "fs";
import { fitTemperature, type FitSample } from "../src/lib/calibration";

const inputPath = process.argv[2];
if (!inputPath) {
  console.error("Usage: npx tsx scripts/calibrate.ts <heldout.jsonl>");
  process.exit(1);
}

const samples: FitSample[] = readFileSync(inputPath, "utf8")
  .split(/\r?\n/)
  .filter((l) => l.trim() && !l.startsWith("//"))
  .map((l, i) => {
    try {
      const row = JSON.parse(l) as { probs: number[]; correctIdx: number };
      if (!Array.isArray(row.probs) || typeof row.correctIdx !== "number") {
        throw new Error("row must be { probs: number[], correctIdx: number }");
      }
      return { probs: row.probs, correctIdx: row.correctIdx };
    } catch (e) {
      throw new Error(`Invalid JSON on line ${i + 1}: ${(e as Error).message}`);
    }
  });

if (!samples.length) {
  console.error("No samples found in dataset");
  process.exit(1);
}

const before = fitTemperature(samples, { minT: 1, maxT: 1 });
const fit = fitTemperature(samples);

console.log("=== OpenJev calibration fit ===");
console.log(`samples: ${samples.length}`);
console.log(`best temperature: ${fit.temperature}`);
console.log(`NLL:   ${before.nll.toFixed(4)} -> ${fit.nll.toFixed(4)}`);
console.log(`Brier: ${before.brier.toFixed(4)} -> ${fit.brier.toFixed(4)}`);
console.log(`ECE:   ${before.ece.toFixed(4)} -> ${fit.ece.toFixed(4)}`);
console.log(
  `\nSet OPENJEV_CALIBRATION_T=${fit.temperature} in .env (or pass calibration_temperature per request).`
);
