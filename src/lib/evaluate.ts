/**
 * OpenJev evaluate backends:
 * - parallel / oneshot: OpenAI-compatible chat LLMs
 * - decider: Mapika/decider HTTP (POST /v1/systemone) — real System One weights
 *
 * Parallel mode defaults to a single-call logprob readout (one completion per
 * question, option scores from top-logprobs). Questions whose primary readout
 * fails degrade individually to per-option micro-scorers, so one unsupported
 * option no longer fails the whole request; a request where every question
 * fails still errors out. Set OPENJEV_PARALLEL_MODE=micro (or readout:
 * "micro" per request) to prefer the legacy micro-scorers. Outbound calls are
 * bounded by OPENJEV_CONCURRENCY (default 6 in flight).
 *
 * Distributions are temperature-scaled with a calibration temperature
 * (OPENJEV_CALIBRATION_T or calibration_temperature per request; default 1 =
 * no rescaling). Fit T on held-out data with `npm run calibrate`.
 */
import OpenAI from "openai";
import {
  buildOptionLetterPrompt,
  parseLetterLogprobs,
  parseYesNoLogprobs,
  DEFAULT_LOGPROB_TOP_K,
  type LetterPrompt,
} from "./logprob";
import {
  resolveCalibrationT,
  softmaxFromLogits,
  temperatureScaleProbs,
} from "./calibration";
import { withRetry } from "./retry";
import { limitConcurrency } from "./concurrency";
import type {
  Answer,
  BackendMode,
  ChoiceAnswer,
  EvaluateRequest,
  EvaluateResponse,
  NoulAnswer,
  Question,
  Questions,
  ScoreAnswer,
} from "./types";

function makeClient(apiKey?: string, baseUrl?: string) {
  const key =
    (apiKey && apiKey.trim()) ||
    process.env.OPENAI_API_KEY ||
    process.env.OPENJEV_API_KEY ||
    process.env.CEREBRAS_API_KEY ||
    process.env.PUTER_API_KEY ||
    process.env.GROQ_API_KEY ||
    process.env.API_KEY;
  if (!key) {
    throw new Error(
      "No API key. Set OPENAI_API_KEY in .env or pass api_key in the request."
    );
  }
  const timeoutRaw = Number(process.env.OPENJEV_TIMEOUT_MS);
  const timeout = Number.isFinite(timeoutRaw) && timeoutRaw > 0 ? timeoutRaw : 30_000;
  // Retries are owned by our own withRetry helper (exponential backoff +
  // Retry-After), so disable the SDK's internal retry loop.
  return new OpenAI({
    apiKey: key,
    baseURL:
      (baseUrl && baseUrl.trim()) ||
      process.env.OPENAI_BASE_URL ||
      process.env.OPENJEV_BASE_URL ||
      undefined,
    timeout,
    maxRetries: 0,
  });
}

function perRequestConcurrency(): number {
  const raw = Number(process.env.OPENJEV_CONCURRENCY);
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : 6;
}

/** Uniform-probability placeholder used when every readout failed for one question. */
function neutralChoiceAnswer(keys: string[]): ChoiceAnswer {
  const probabilities: Record<string, number> = {};
  const p = r3(1 / keys.length);
  keys.forEach((k) => {
    probabilities[k] = p;
  });
  return { type: "choice", choice: keys[0], confidence: p, probabilities };
}

function neutralScoreAnswer(levels: string[]): ScoreAnswer {
  const legend: Record<string, string> = {};
  levels.forEach((l, i) => {
    legend[String(i)] = l;
  });
  const p = r3(1 / levels.length);
  const probabilities: Record<string, number> = {};
  levels.forEach((_l, i) => {
    probabilities[String(i)] = p;
  });
  return {
    type: "score",
    score: Math.round(((levels.length - 1) / 2) * 100) / 100,
    confidence: p,
    legend,
    probabilities,
  };
}

interface ScoreCallResult {
  probability: number;
  tokens_in: number;
  tokens_out: number;
  retries: number;
}

function r3(x: number): number {
  return Math.round(x * 1000) / 1000;
}

function confidenceFromProbs(probs: number[]): number {
  const sorted = [...probs].sort((a, b) => b - a);
  const gap = (sorted[0] ?? 0) - (sorted[1] ?? 0);
  return Math.min(1, Math.max(0, r3(0.5 * (sorted[0] ?? 0) + 0.5 * (gap + 0.5))));
}

async function scoreProposition(
  client: OpenAI,
  model: string,
  state: string,
  statement: string,
  temperature: number,
  signal?: AbortSignal
): Promise<ScoreCallResult> {
  const system =
    "You are a calibrated probability estimator. " +
    "Given a STATE and a STATEMENT, return only how likely the statement is true " +
    "based solely on the state. Do not invent facts. " +
    'Respond with JSON: {"p": <number 0 to 1>}.';
  const user = `STATE:\n${state}\n\nSTATEMENT:\n${statement}\n\nHow likely is the statement true (0–1)?`;

  const create = (
    responseFormat:
      | OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming["response_format"]
      | undefined
  ) =>
    client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens: 32,
      response_format: responseFormat,
    } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming, { signal });

  // json_schema first (strict providers), json_object as a fallback shape.
  const { result: completion, retries } = await withRetry("micro-score", async () => {
    try {
      return await create({
        type: "json_schema",
        json_schema: {
          name: "prob",
          strict: true,
          schema: {
            type: "object",
            properties: { p: { type: "number", minimum: 0, maximum: 1 } },
            required: ["p"],
            additionalProperties: false,
          },
        },
      });
    } catch {
      return await create({ type: "json_object" });
    }
  });

  const raw = completion.choices[0]?.message?.content ?? "{}";
  let p = 0.5;
  try {
    const parsed = JSON.parse(raw) as { p?: number; probability?: number };
    p = Number(parsed.p ?? parsed.probability ?? 0.5);
    if (!Number.isFinite(p)) p = 0.5;
    p = Math.min(1, Math.max(0, p));
  } catch {
    const m = raw.match(/0?\.\d+|[01](?:\.0+)?/);
    if (m) p = Math.min(1, Math.max(0, parseFloat(m[0])));
  }

  return {
    probability: p,
    tokens_in: completion.usage?.prompt_tokens ?? 0,
    tokens_out: completion.usage?.completion_tokens ?? 0,
    retries,
  };
}

function scoresToDistribution(scores: number[], calibrationT = 1): number[] {
  const logits = scores.map((p) => {
    const clamped = Math.min(0.999, Math.max(0.001, p));
    return Math.log(clamped / (1 - clamped));
  });
  return softmaxFromLogits(logits, calibrationT);
}

// ── logprob readout (one call per question) ────────────────────────────────

interface LetterReadResult {
  logits: number[];
  matched: number;
  tokens_in: number;
  tokens_out: number;
  retries: number;
}

async function readLetterLogprobs(
  client: OpenAI,
  model: string,
  prompt: LetterPrompt,
  optionCount: number,
  optionKeys?: string[],
  signal?: AbortSignal
): Promise<LetterReadResult> {
  const { result: completion, retries } = await withRetry("logprob-read", () =>
    client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: prompt.system },
        { role: "user", content: prompt.user },
      ],
      temperature: 0,
      max_tokens: 4,
      logprobs: true,
      top_logprobs: DEFAULT_LOGPROB_TOP_K,
    } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
      { signal })
  );

  const top = completion.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs ?? [];
  const parsed = parseLetterLogprobs(
    top.map((t) => ({ token: t.token, logprob: t.logprob })),
    optionCount,
    optionKeys
  );
  if (parsed.matched === 0) {
    throw new Error(
      "logprob readout: provider returned no usable option logprobs (does it support logprobs?)"
    );
  }
  return {
    logits: parsed.logits,
    matched: parsed.matched,
    tokens_in: completion.usage?.prompt_tokens ?? 0,
    tokens_out: completion.usage?.completion_tokens ?? 0,
    retries,
  };
}

async function evaluateChoiceLogprob(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "choice" }>,
  calibrationT: number,
  signal?: AbortSignal
) {
  const keys = Object.keys(q.criteria);
  const instructions = q.instructions ?? `Select the best label for "${name}"`;
  const options = keys.map((k) => ({ key: k, description: q.criteria[k] }));
  const prompt = buildOptionLetterPrompt(state, instructions, options);
  const read = await readLetterLogprobs(client, model, prompt, keys.length, keys, signal);
  const probs = softmaxFromLogits(read.logits, calibrationT);

  const probabilities: Record<string, number> = {};
  keys.forEach((k, i) => {
    probabilities[k] = r3(probs[i]);
  });
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;

  return {
    answer: {
      type: "choice" as const,
      choice: keys[bestIdx],
      confidence: confidenceFromProbs(probs),
      probabilities,
    } satisfies ChoiceAnswer,
    tokens_in: read.tokens_in,
    tokens_out: read.tokens_out,
    retries: read.retries,
    calls: 1,
    matched: read.matched,
  };
}

async function evaluateScoreLogprob(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "score" }>,
  calibrationT: number,
  signal?: AbortSignal
) {
  const levels = q.criteria;
  const instructions = q.instructions ?? `Rate "${name}"`;
  const prompt = buildOptionLetterPrompt(
    state,
    `${instructions} (levels: ${levels.map((l, i) => `${i}=${l}`).join(", ")})`,
    levels.map((l) => ({ key: l }))
  );
  const read = await readLetterLogprobs(client, model, prompt, levels.length, undefined, signal);
  const probs = softmaxFromLogits(read.logits, calibrationT);

  let score = 0;
  probs.forEach((p, i) => {
    score += p * i;
  });
  const legend: Record<string, string> = {};
  levels.forEach((l, i) => {
    legend[String(i)] = l;
  });
  const probabilityMap: Record<string, number> = {};
  probs.forEach((p, i) => {
    probabilityMap[String(i)] = r3(p);
  });

  return {
    answer: {
      type: "score" as const,
      score: Math.round(score * 100) / 100,
      confidence: confidenceFromProbs(probs),
      legend,
      probabilities: probabilityMap,
    } satisfies ScoreAnswer,
    tokens_in: read.tokens_in,
    tokens_out: read.tokens_out,
    retries: read.retries,
    calls: 1,
    matched: read.matched,
  };
}

async function evaluateNoulLogprob(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "noul" | "boolean" }>,
  calibrationT: number,
  signal?: AbortSignal
) {
  let statement = q.instructions?.trim() || `The proposition "${name}" is true.`;
  if (!statement.endsWith("?") && !statement.endsWith(".")) statement += ".";

  // Yes/No tokens usually appear directly in the top-logprobs window; if the
  // model answered with a letter instead, fall back to A=Yes / B=No parsing.
  const prompt = buildOptionLetterPrompt(state, statement, [
    { key: "Yes" },
    { key: "No" },
  ]);
  const { result: read, retries: noulRetries } = await withRetry("logprob-read", () =>
    client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: prompt.system },
        { role: "user", content: prompt.user },
      ],
      temperature: 0,
      max_tokens: 4,
      logprobs: true,
      top_logprobs: DEFAULT_LOGPROB_TOP_K,
    } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming,
      { signal })
  );
  const top = read.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs ?? [];
  const yn = parseYesNoLogprobs(
    top.map((t) => ({ token: t.token, logprob: t.logprob }))
  );

  let yesLogit: number;
  let noLogit: number;
  let matched: number;
  if (yn.yesLogit === null && yn.noLogit === null) {
    const parsed = parseLetterLogprobs(
      top.map((t) => ({ token: t.token, logprob: t.logprob })),
      2
    );
    if (parsed.matched === 0) {
      throw new Error("logprob readout: no yes/no or A/B tokens in response");
    }
    yesLogit = parsed.logits[0];
    noLogit = parsed.logits[1];
    matched = parsed.matched;
  } else {
    const floor = Math.min(yn.yesLogit ?? 0, yn.noLogit ?? 0) - 5;
    yesLogit = yn.yesLogit ?? floor;
    noLogit = yn.noLogit ?? floor;
    matched = (yn.yesLogit !== null ? 1 : 0) + (yn.noLogit !== null ? 1 : 0);
  }

  const probs = softmaxFromLogits([yesLogit, noLogit], calibrationT);
  return {
    answer: { type: "noul" as const, noul: r3(probs[0]) } satisfies NoulAnswer,
    tokens_in: read.usage?.prompt_tokens ?? 0,
    tokens_out: read.usage?.completion_tokens ?? 0,
    retries: noulRetries,
    calls: 1,
    matched,
  };
}

// ── micro-scorer readout (one call per option) ─────────────────────────────

async function evaluateChoiceParallel(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "choice" }>,
  temperature: number,
  calibrationT: number,
  signal?: AbortSignal
) {
  const keys = Object.keys(q.criteria);
  const instructions = q.instructions ?? `Select the best label for "${name}"`;
  const results = await Promise.all(
    keys.map((key) => {
      const desc = q.criteria[key];
      const statement =
        `The correct ${instructions.toLowerCase().replace(/\?$/, "")} is "${key}"` +
        (desc ? ` (${desc})` : "") +
        ".";
      return scoreProposition(client, model, state, statement, temperature, signal);
    })
  );
  const probs = scoresToDistribution(
    results.map((r) => r.probability),
    calibrationT
  );
  const probabilities: Record<string, number> = {};
  keys.forEach((k, i) => {
    probabilities[k] = r3(probs[i]);
  });
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;
  return {
    answer: {
      type: "choice" as const,
      choice: keys[bestIdx],
      confidence: confidenceFromProbs(probs),
      probabilities,
    } satisfies ChoiceAnswer,
    tokens_in: results.reduce((s, r) => s + r.tokens_in, 0),
    tokens_out: results.reduce((s, r) => s + r.tokens_out, 0),
    retries: results.reduce((s, r) => s + r.retries, 0),
    calls: keys.length,
  };
}

async function evaluateScoreParallel(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "score" }>,
  temperature: number,
  calibrationT: number,
  signal?: AbortSignal
) {
  const levels = q.criteria;
  const instructions = q.instructions ?? `Rate "${name}"`;
  const results = await Promise.all(
    levels.map((label, i) => {
      const statement = `On the scale for "${instructions}", the most appropriate rating is level ${i}: "${label}".`;
      return scoreProposition(client, model, state, statement, temperature, signal);
    })
  );
  const probs = scoresToDistribution(
    results.map((r) => r.probability),
    calibrationT
  );
  let score = 0;
  probs.forEach((p, i) => {
    score += p * i;
  });
  score = Math.round(score * 100) / 100;
  const legend: Record<string, string> = {};
  levels.forEach((l, i) => {
    legend[String(i)] = l;
  });
  const probabilityMap: Record<string, number> = {};
  probs.forEach((p, i) => {
    probabilityMap[String(i)] = r3(p);
  });
  return {
    answer: {
      type: "score" as const,
      score,
      confidence: confidenceFromProbs(probs),
      legend,
      probabilities: probabilityMap,
    } satisfies ScoreAnswer,
    tokens_in: results.reduce((s, r) => s + r.tokens_in, 0),
    tokens_out: results.reduce((s, r) => s + r.tokens_out, 0),
    retries: results.reduce((s, r) => s + r.retries, 0),
    calls: levels.length,
  };
}

async function evaluateNoulParallel(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "noul" | "boolean" }>,
  temperature: number,
  signal?: AbortSignal
) {
  let statement = q.instructions?.trim() || `The proposition "${name}" is true.`;
  if (!statement.endsWith("?") && !statement.endsWith(".")) statement += ".";
  const result = await scoreProposition(client, model, state, statement, temperature, signal);
  return {
    answer: { type: "noul" as const, noul: r3(result.probability) } satisfies NoulAnswer,
    tokens_in: result.tokens_in,
    tokens_out: result.tokens_out,
    retries: result.retries,
    calls: 1,
  };
}

// ── parallel dispatch: per-question readout with degradation ───────────────

interface ParallelOutcome {
  answers: Record<string, Answer>;
  tokens_in: number;
  tokens_out: number;
  calls: number;
  retries: number;
  readout: "logprob" | "micro";
  matchedLetters?: number;
  /** Questions that had to use their second-choice readout. */
  fallbackQuestions: number;
  /** Questions no readout could answer (they carry neutral placeholders). */
  failedQuestions: string[];
}

type ParallelReadout = "logprob" | "micro";

interface QuestionOutcome {
  answer: Answer;
  tokens_in: number;
  tokens_out: number;
  retries: number;
  calls: number;
  readout?: ParallelReadout;
  matched?: number;
}

async function readQuestionOnce(
  client: OpenAI,
  model: string,
  stateStr: string,
  name: string,
  q: Question,
  temperature: number,
  calibrationT: number,
  readout: ParallelReadout,
  signal?: AbortSignal
): Promise<QuestionOutcome> {
  if (readout === "logprob") {
    if (q.type === "choice") {
      return await evaluateChoiceLogprob(client, model, stateStr, name, q, calibrationT, signal);
    }
    if (q.type === "score") {
      return await evaluateScoreLogprob(client, model, stateStr, name, q, calibrationT, signal);
    }
    return await evaluateNoulLogprob(client, model, stateStr, name, q, calibrationT, signal);
  }
  if (q.type === "choice") {
    return await evaluateChoiceParallel(
      client,
      model,
      stateStr,
      name,
      q,
      temperature,
      calibrationT,
      signal
    );
  }
  if (q.type === "score") {
    return await evaluateScoreParallel(
      client,
      model,
      stateStr,
      name,
      q,
      temperature,
      calibrationT,
      signal
    );
  }
  return await evaluateNoulParallel(client, model, stateStr, name, q, temperature, signal);
}

/**
 * Run one question through its readout chain: the requested readout first,
 * then the other one. Throws only when every readout for this question fails.
 */
async function evaluateQuestionWithFallback(
  client: OpenAI,
  model: string,
  stateStr: string,
  name: string,
  q: Question,
  temperature: number,
  calibrationT: number,
  requested: ParallelReadout,
  signal?: AbortSignal
): Promise<QuestionOutcome & { degraded: boolean }> {
  const order: ParallelReadout[] = requested === "micro" ? ["micro"] : ["logprob", "micro"];
  let lastErr: unknown;
  for (let i = 0; i < order.length; i++) {
    const readout = order[i];
    try {
      const out = await readQuestionOnce(
        client,
        model,
        stateStr,
        name,
        q,
        temperature,
        calibrationT,
        readout,
        signal
      );
      return { ...out, readout, degraded: i > 0 };
    } catch (err) {
      lastErr = err;
      console.warn(
        `[openjev] question "${name}" via ${readout} failed: ${
          err instanceof Error ? err.message : String(err)
        }${i + 1 < order.length ? ` — degrading to ${order[i + 1]}` : ""}`
      );
    }
  }
  throw lastErr;
}

/** Uniform-probability answer of the right shape, used when all readouts fail. */
function neutralAnswerFor(q: Question): Answer {
  if (q.type === "choice") return neutralChoiceAnswer(Object.keys(q.criteria));
  if (q.type === "score") return neutralScoreAnswer(q.criteria);
  return { type: "noul", noul: 0.5 };
}

async function evaluateParallelQuestions(
  client: OpenAI,
  model: string,
  stateStr: string,
  questions: Questions,
  temperature: number,
  calibrationT: number,
  requested: ParallelReadout,
  signal?: AbortSignal
): Promise<ParallelOutcome> {
  const entries = Object.entries(questions);
  const settled = await limitConcurrency(
    entries,
    perRequestConcurrency(),
    async ([name, q]) => {
      try {
        return {
          name,
          ok: true as const,
          ...(await evaluateQuestionWithFallback(
            client,
            model,
            stateStr,
            name,
            q,
            temperature,
            calibrationT,
            requested,
            signal
          )),
        };
      } catch (err) {
        return { name, ok: false as const, err, answer: neutralAnswerFor(q) };
      }
    }
  );

  const failed = settled.filter((s) => !s.ok);
  if (failed.length === settled.length && settled.length > 0) {
    const first = failed[0]?.err;
    throw new Error(
      `all ${settled.length} question(s) failed: ${
        first instanceof Error ? first.message : String(first)
      }`
    );
  }

  const answers: Record<string, Answer> = {};
  let tokens_in = 0;
  let tokens_out = 0;
  let calls = 0;
  let retries = 0;
  let matchedLetters: number | undefined;
  let fallbackQuestions = 0;
  const failedQuestions: string[] = [];
  let anyDegraded = false;
  let allMicro = true;
  for (const s of settled) {
    answers[s.name] = s.answer;
    if (!s.ok) {
      failedQuestions.push(s.name);
      continue; // neutral placeholder — no tokens or calls to report
    }
    if (s.degraded) fallbackQuestions++;
    anyDegraded = anyDegraded || s.degraded;
    allMicro = allMicro && s.readout === "micro";
    tokens_in += s.tokens_in;
    tokens_out += s.tokens_out;
    calls += s.calls;
    retries += s.retries;
    if (s.readout === "logprob" && typeof s.matched === "number") {
      matchedLetters = (matchedLetters ?? 0) + s.matched;
    }
  }
  // Report the readout that actually produced the answers: when every
  // question had to degrade, "micro" is the honest label; mixed batches keep
  // the requested readout with fallback_questions conveying the split.
  const effectiveReadout: ParallelReadout =
    allMicro && anyDegraded ? "micro" : requested;
  return {
    answers,
    tokens_in,
    tokens_out,
    calls,
    retries,
    readout: effectiveReadout,
    matchedLetters,
    fallbackQuestions,
    failedQuestions,
  };
}

// ── oneshot backend ────────────────────────────────────────────────────────

async function evaluateOneshot(
  client: OpenAI,
  model: string,
  state: string,
  questions: Questions,
  temperature: number,
  signal?: AbortSignal
) {
  const questionDescriptions: string[] = [];
  for (const [name, q] of Object.entries(questions)) {
    let desc = `- ${name} (${q.type}): ${q.instructions ?? ""}`;
    if (q.type === "choice") {
      desc += `\n  Options → ${Object.entries(q.criteria)
        .map(([k, v]) => `${k}: ${v}`)
        .join(", ")}`;
    } else if (q.type === "score") {
      desc += `\n  Levels → ${q.criteria.map((l, i) => `${i}=${l}`).join(" | ")}`;
    }
    questionDescriptions.push(desc);
  }
  const system =
    "You are a precise decision engine. Answer every question based only on the provided state. " +
    "Return probabilities that reflect genuine uncertainty. Do not invent information.";
  const user =
    `STATE:\n${state}\n\nQUESTIONS:\n${questionDescriptions.join("\n")}\n\n` +
    `Respond with JSON. For choice: { "choice", "confidence", "probabilities" }. ` +
    `For score: { "score", "confidence" }. For noul: { "noul" }. Top-level keys = question names.`;

  const completion = await client.chat.completions.create({
    model,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    temperature,
    response_format: { type: "json_object" },
  } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming, { signal });

  const content = completion.choices[0]?.message?.content ?? "{}";
  let parsed: Record<string, Record<string, unknown>> = {};
  try {
    parsed = JSON.parse(content);
  } catch {
    throw new Error(`One-shot model returned non-JSON: ${content.slice(0, 200)}`);
  }

  const answers: Record<string, Answer> = {};
  for (const [name, q] of Object.entries(questions)) {
    const data = parsed[name] ?? {};
    if (q.type === "choice") {
      const choice = String(data.choice ?? Object.keys(q.criteria)[0]);
      const confidence = Number(data.confidence ?? 0.5);
      const probabilities = (data.probabilities as Record<string, number>) ?? {
        [choice]: confidence,
      };
      for (const k of Object.keys(q.criteria)) {
        if (probabilities[k] === undefined) probabilities[k] = 0;
      }
      answers[name] = { type: "choice", choice, confidence, probabilities };
    } else if (q.type === "score") {
      const legend: Record<string, string> = {};
      q.criteria.forEach((l, i) => {
        legend[String(i)] = l;
      });
      answers[name] = {
        type: "score",
        score: Number(data.score ?? 0),
        confidence: Number(data.confidence ?? 0.5),
        legend,
      };
    } else {
      answers[name] = {
        type: "noul",
        noul: Number(data.noul ?? data.probability ?? 0.5),
      };
    }
  }

  return {
    answers,
    tokens_in: completion.usage?.prompt_tokens ?? 0,
    tokens_out: completion.usage?.completion_tokens ?? 0,
    calls: 1,
  };
}

/** Temperature-scale the choice probabilities returned by the oneshot backend. */
function calibrateOneshotAnswer(
  answer: Answer,
  calibrationT: number
): Answer {
  if (calibrationT === 1) return answer;
  if (answer.type === "choice" && answer.probabilities) {
    const keys = Object.keys(answer.probabilities);
    const scaled = temperatureScaleProbs(
      keys.map((k) => answer.probabilities![k]),
      calibrationT
    );
    const probabilities: Record<string, number> = {};
    keys.forEach((k, i) => {
      probabilities[k] = r3(scaled[i]);
    });
    return { ...answer, probabilities, confidence: confidenceFromProbs(scaled) };
  }
  if (answer.type === "score" && answer.probabilities) {
    const keys = Object.keys(answer.probabilities);
    const scaled = temperatureScaleProbs(
      keys.map((k) => answer.probabilities![k]),
      calibrationT
    );
    const probabilities: Record<string, number> = {};
    keys.forEach((k, i) => {
      probabilities[k] = r3(scaled[i]);
    });
    return { ...answer, probabilities, confidence: confidenceFromProbs(scaled) };
  }
  return answer;
}

// ── decider backend (unchanged wire format) ────────────────────────────────

/**
 * Call Mapika/decider HTTP server (TypeSafe wire format).
 * https://github.com/Mapika/decider — scripts/serve.sh Mapika/decider-2b 8000
 */
async function evaluateDecider(
  state: string | Record<string, unknown> | unknown[],
  questions: Questions,
  deciderUrl: string,
  apiKey?: string,
  signal?: AbortSignal
): Promise<{
  answers: Record<string, Answer>;
  model: string;
  tokens_in: number | null;
  tokens_out: number | null;
}> {
  const base = deciderUrl.replace(/\/$/, "");
  const url = `${base}/v1/systemone`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const key =
    (apiKey && apiKey.trim()) ||
    process.env.DECIDER_API_KEY ||
    process.env.TYPESAFE_API_KEY ||
    "local";
  headers.Authorization = `Bearer ${key}`;

  const body = {
    state,
    questions,
    model: "decider",
  };

  const { result: res } = await withRetry("decider", () =>
    fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: signal
        ? AbortSignal.any([signal, AbortSignal.timeout(30_000)])
        : AbortSignal.timeout(30_000),
    })
  );

  const text = await res.text();
  if (!res.ok) {
    throw new Error(
      `decider ${res.status} from ${url}: ${text.slice(0, 400)}. ` +
        `Is the server running? scripts/serve.sh Mapika/decider-2b 8000`
    );
  }

  let data: {
    model?: string;
    answers?: Record<string, Record<string, unknown>>;
    usage?: { input_tokens?: number; output_tokens?: number };
  };
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`decider returned non-JSON: ${text.slice(0, 200)}`);
  }

  const answers: Record<string, Answer> = {};
  const rawAnswers = data.answers ?? {};

  for (const [name, q] of Object.entries(questions)) {
    const a = rawAnswers[name] ?? {};
    if (q.type === "choice") {
      const probs = (a.probabilities as Record<string, number>) ?? {};
      const choice = String(a.choice ?? Object.keys(q.criteria)[0] ?? "");
      answers[name] = {
        type: "choice",
        choice,
        confidence: Number(a.confidence ?? a.certainty ?? probs[choice] ?? 0.5),
        probabilities: Object.keys(q.criteria).reduce(
          (acc, k) => {
            acc[k] = Number(probs[k] ?? 0);
            return acc;
          },
          {} as Record<string, number>
        ),
      };
    } else if (q.type === "score") {
      const legend: Record<string, string> = {};
      q.criteria.forEach((l, i) => {
        legend[String(i)] = l;
      });
      const probs =
        (a.probabilities as Record<string, number>) ??
        (a.level_fit as Record<string, number>) ??
        {};
      answers[name] = {
        type: "score",
        score: Number(a.score ?? 0),
        confidence: Number(a.confidence ?? a.fit_mass ?? 0.5),
        legend,
        probabilities: probs,
      };
    } else {
      answers[name] = {
        type: "noul",
        noul: Number(a.noul ?? a.probability ?? 0.5),
      };
    }
  }

  return {
    answers,
    model: data.model ?? "Mapika/decider-2b",
    tokens_in: data.usage?.input_tokens ?? null,
    tokens_out: data.usage?.output_tokens ?? null,
  };
}

// ── entrypoint ─────────────────────────────────────────────────────────────

export async function evaluate(req: EvaluateRequest): Promise<EvaluateResponse> {
  const t0 = Date.now();
  const {
    state,
    questions,
    model = process.env.OPENJEV_MODEL || "gpt-4o-mini",
    base_url,
    api_key,
    temperature = 0,
    mode = "parallel",
    decider_url,
    calibration_temperature,
    readout,
    signal,
  } = req;

  if (!questions || Object.keys(questions).length === 0) {
    throw new Error("At least one question is required");
  }

  // ── decider backend (Mapika/decider) ─────────────────────────────────
  if (mode === "decider") {
    const url =
      (decider_url && decider_url.trim()) ||
      process.env.DECIDER_BASE_URL ||
      "http://localhost:8000";

    const result = await evaluateDecider(state, questions, url, api_key, signal);

    return {
      model: result.model,
      answers: result.answers,
      usage: {
        input_tokens: result.tokens_in,
        output_tokens: result.tokens_out,
      },
      meta: {
        mode: "decider",
        latency_ms: Date.now() - t0,
        parallel_calls: 1,
        backend: url,
      },
    };
  }

  const stateStr =
    typeof state === "string" ? state : JSON.stringify(state, null, 2);
  const client = makeClient(api_key, base_url);
  const calibrationT = resolveCalibrationT(
    calibration_temperature,
    process.env.OPENJEV_CALIBRATION_T
  );

  if (mode === "parallel") {
    const requested: ParallelReadout =
      readout ?? (process.env.OPENJEV_PARALLEL_MODE as ParallelReadout) ?? "logprob";

    const outcome = await evaluateParallelQuestions(
      client,
      model,
      stateStr,
      questions,
      temperature,
      calibrationT,
      requested,
      signal
    );

    return {
      model,
      answers: outcome.answers,
      usage: {
        input_tokens: outcome.tokens_in || null,
        output_tokens: outcome.tokens_out || null,
      },
      meta: {
        mode: "parallel",
        latency_ms: Date.now() - t0,
        parallel_calls: outcome.calls,
        backend: "openai-compatible",
        readout: outcome.readout,
        calibration_t: calibrationT,
        fallback_used:
          outcome.fallbackQuestions > 0 ? "per-question degradation" : null,
        fallback_questions: outcome.fallbackQuestions > 0 ? outcome.fallbackQuestions : undefined,
        failed_questions: outcome.failedQuestions.length
          ? outcome.failedQuestions
          : undefined,
        retries: outcome.retries,
        ...(outcome.matchedLetters !== undefined
          ? { matched_letters: outcome.matchedLetters }
          : {}),
      },
    };
  }

  const { result: oneshot, retries } = await withRetry("oneshot", () =>
    evaluateOneshot(client, model, stateStr, questions, temperature, signal)
  );
  const answers: Record<string, Answer> = {};
  for (const [name, answer] of Object.entries(oneshot.answers)) {
    answers[name] = calibrateOneshotAnswer(answer, calibrationT);
  }

  return {
    model,
    answers,
    usage: {
      input_tokens: oneshot.tokens_in || null,
      output_tokens: oneshot.tokens_out || null,
    },
    meta: {
      mode: "oneshot",
      latency_ms: Date.now() - t0,
      parallel_calls: oneshot.calls,
      backend: "openai-compatible",
      readout: "json",
      calibration_t: calibrationT,
      retries,
    },
  };
}
