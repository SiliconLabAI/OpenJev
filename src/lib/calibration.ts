/**
 * Calibration utilities for the softmax distributions OpenJev produces.
 *
 * Temperature scaling (Guo et al. 2017) divides logits by T before the
 * softmax: T > 1 flattens over-confident distributions, T < 1 sharpens
 * under-confident ones. Fit T on a held-out dataset with
 * `npm run calibrate -- data.jsonl` (minimizes mean NLL), then set
 * OPENJEV_CALIBRATION_T in the environment or pass
 * `calibration_temperature` per request.
 */

const EPS = 1e-12;

export function softmaxFromLogits(logits: number[], temperature = 1): number[] {
  const t = Math.max(0.05, temperature);
  const scaled = logits.map((x) => x / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

/**
 * Apply temperature scaling directly to a probability distribution.
 * Recovers logits as log(p) (identical up to the additive constant the
 * softmax ignores), then re-softmaxes with temperature T.
 */
export function temperatureScaleProbs(probs: number[], temperature: number): number[] {
  if (temperature === 1) return [...probs];
  const logits = probs.map((p) => Math.log(Math.max(EPS, Math.min(1 - EPS, p))));
  return softmaxFromLogits(logits, temperature);
}

/**
 * Resolve the calibration temperature from (in order of precedence):
 * explicit request value, env value, default 1 (no rescaling).
 */
export function resolveCalibrationT(requested?: number | null, envValue?: string): number {
  const raw = requested ?? (envValue !== undefined && envValue !== "" ? Number(envValue) : NaN);
  const t = Number(raw);
  if (!Number.isFinite(t) || t <= 0) return 1;
  return Math.min(10, Math.max(0.1, t));
}

/** Multiclass Brier score: sum over classes of (p_i - y_i)^2. Range 0 (perfect) to 2. */
export function brierScore(probs: number[], correctIdx: number): number {
  let score = 0;
  probs.forEach((p, i) => {
    const y = i === correctIdx ? 1 : 0;
    score += (p - y) ** 2;
  });
  return score;
}

export interface FitSample {
  probs: number[];
  correctIdx: number;
}

/**
 * Expected Calibration Error over samples, bucketed by confidence
 * (max probability) into `bins` equal-width buckets.
 */
export function expectedCalibrationError(samples: FitSample[], bins = 10): number {
  const n = samples.length;
  if (!n) return 0;
  const bucketN = new Array<number>(bins).fill(0);
  const bucketConf = new Array<number>(bins).fill(0);
  const bucketAcc = new Array<number>(bins).fill(0);
  for (const s of samples) {
    const conf = Math.max(...s.probs);
    const bin = Math.min(bins - 1, Math.max(0, Math.floor(conf * bins)));
    bucketN[bin] += 1;
    bucketConf[bin] += conf;
    if (argmax(s.probs) === s.correctIdx) bucketAcc[bin] += 1;
  }
  let ece = 0;
  for (let b = 0; b < bins; b++) {
    if (!bucketN[b]) continue;
    ece += (bucketN[b] / n) * Math.abs(bucketAcc[b] / bucketN[b] - bucketConf[b] / bucketN[b]);
  }
  return ece;
}

export function argmax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function meanNll(samples: FitSample[], t: number): number {
  if (!samples.length) return 0;
  let nll = 0;
  for (const s of samples) {
    const scaled = temperatureScaleProbs(s.probs, t);
    nll += -Math.log(Math.max(EPS, scaled[s.correctIdx]));
  }
  return nll / samples.length;
}

function meanBrier(samples: FitSample[], t: number): number {
  if (!samples.length) return 0;
  return (
    samples.reduce((acc, s) => acc + brierScore(temperatureScaleProbs(s.probs, t), s.correctIdx), 0) /
    samples.length
  );
}

export interface FitResult {
  temperature: number;
  nll: number;
  brier: number;
  ece: number;
  grid: { t: number; nll: number; brier: number }[];
}

/**
 * Grid-search the temperature that minimizes mean NLL over held-out samples
 * (the standard temperature-scaling objective). Reports Brier and ECE at
 * the chosen T as well.
 */
export function fitTemperature(
  samples: FitSample[],
  opts?: { minT?: number; maxT?: number; step?: number }
): FitResult {
  const minT = opts?.minT ?? 0.5;
  const maxT = opts?.maxT ?? 3;
  const step = opts?.step ?? 0.05;
  const grid: { t: number; nll: number; brier: number }[] = [];
  for (let t = minT; t <= maxT + 1e-9; t += step) {
    const tr = Math.round(t * 100) / 100;
    grid.push({ t: tr, nll: meanNll(samples, tr), brier: meanBrier(samples, tr) });
  }
  const best = grid.reduce((a, b) => (b.nll < a.nll ? b : a));
  const scaled = samples.map((s) => ({ probs: temperatureScaleProbs(s.probs, best.t), correctIdx: s.correctIdx }));
  return {
    temperature: best.t,
    nll: best.nll,
    brier: best.brier,
    ece: expectedCalibrationError(scaled),
    grid,
  };
}
