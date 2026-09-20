/**
 * OpenJev — open approximation of TypeSafe Jev's System One contract.
 *
 * Design (inspired by how Jev actually works):
 *
 * 1. Fixed answer space — no free-form text generation.
 * 2. Parallel sampler — each candidate (choice key, score level, or noul)
 *    is scored independently against the same state, then normalized.
 * 3. Questions are also evaluated in parallel (Promise.all).
 *
 * This avoids the flaky single-shot "return a big JSON blob" failure mode.
 */
import OpenAI from "openai";
import type {
  Answer,
  ChoiceAnswer,
  EvaluateRequest,
  EvaluateResponse,
  NoulAnswer,
  Question,
  Questions,
  ScoreAnswer,
} from "./types";

function makeClient(apiKey?: string, baseUrl?: string) {
  return new OpenAI({
    apiKey:
      apiKey ||
      process.env.OPENAI_API_KEY ||
      process.env.CEREBRAS_API_KEY ||
      process.env.PUTER_API_KEY ||
      process.env.GROQ_API_KEY ||
      "sk-placeholder",
    baseURL: baseUrl || undefined,
  });
}

interface ScoreCallResult {
  probability: number;
  tokens_in: number;
  tokens_out: number;
}

/**
 * Ask the model: "Given STATE, how likely is STATEMENT true?"
 * Tiny constrained output — only a probability, no free text.
 */
async function scoreProposition(
  client: OpenAI,
  model: string,
  state: string,
  statement: string,
  temperature: number
): Promise<ScoreCallResult> {
  const system =
    "You are a calibrated probability estimator. " +
    "Given a STATE and a STATEMENT, return only how likely the statement is true " +
    "based solely on the state. Do not invent facts. " +
    'Respond with JSON: {"p": <number 0 to 1>}.';

  const user = `STATE:\n${state}\n\nSTATEMENT:\n${statement}\n\nHow likely is the statement true (0–1)?`;

  let completion: OpenAI.Chat.Completions.ChatCompletion;
  try {
    completion = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens: 32,
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "prob",
          strict: true,
          schema: {
            type: "object",
            properties: {
              p: { type: "number", minimum: 0, maximum: 1 },
            },
            required: ["p"],
            additionalProperties: false,
          },
        },
      },
    });
  } catch {
    completion = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      max_tokens: 32,
      response_format: { type: "json_object" },
    });
  }

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
  };
}

function softmax(logits: number[], temperature = 1): number[] {
  const t = Math.max(0.05, temperature);
  const scaled = logits.map((x) => x / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

/** Independent relevance scores [0,1] → probability distribution via logit + softmax */
function scoresToDistribution(scores: number[]): number[] {
  const logits = scores.map((p) => {
    const clamped = Math.min(0.999, Math.max(0.001, p));
    return Math.log(clamped / (1 - clamped));
  });
  return softmax(logits, 1);
}

async function evaluateChoiceParallel(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "choice" }>,
  temperature: number
): Promise<{ answer: ChoiceAnswer; tokens_in: number; tokens_out: number; calls: number }> {
  const keys = Object.keys(q.criteria);
  const instructions = q.instructions ?? `Select the best label for "${name}"`;

  const results = await Promise.all(
    keys.map((key) => {
      const desc = q.criteria[key];
      const statement =
        `The correct ${instructions.toLowerCase().replace(/\?$/, "")} is "${key}"` +
        (desc ? ` (${desc})` : "") +
        ".";
      return scoreProposition(client, model, state, statement, temperature);
    })
  );

  const rawScores = results.map((r) => r.probability);
  const probs = scoresToDistribution(rawScores);
  const probabilities: Record<string, number> = {};
  keys.forEach((k, i) => {
    probabilities[k] = Math.round(probs[i] * 1000) / 1000;
  });

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }
  const choice = keys[bestIdx];
  const sorted = [...probs].sort((a, b) => b - a);
  const gap = (sorted[0] ?? 0) - (sorted[1] ?? 0);
  const confidence = Math.min(
    1,
    Math.max(0, Math.round((0.5 * (sorted[0] ?? 0) + 0.5 * (gap + 0.5)) * 1000) / 1000)
  );

  return {
    answer: { type: "choice", choice, confidence, probabilities },
    tokens_in: results.reduce((s, r) => s + r.tokens_in, 0),
    tokens_out: results.reduce((s, r) => s + r.tokens_out, 0),
    calls: keys.length,
  };
}

async function evaluateScoreParallel(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "score" }>,
  temperature: number
): Promise<{ answer: ScoreAnswer; tokens_in: number; tokens_out: number; calls: number }> {
  const levels = q.criteria;
  const instructions = q.instructions ?? `Rate "${name}"`;

  const results = await Promise.all(
    levels.map((label, i) => {
      const statement = `On the scale for "${instructions}", the most appropriate rating is level ${i}: "${label}".`;
      return scoreProposition(client, model, state, statement, temperature);
    })
  );

  const rawScores = results.map((r) => r.probability);
  const probs = scoresToDistribution(rawScores);

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
    probabilityMap[String(i)] = Math.round(p * 1000) / 1000;
  });

  const sorted = [...probs].sort((a, b) => b - a);
  const gap = (sorted[0] ?? 0) - (sorted[1] ?? 0);
  const confidence = Math.min(
    1,
    Math.max(0, Math.round((0.5 * (sorted[0] ?? 0) + 0.5 * (gap + 0.5)) * 1000) / 1000)
  );

  return {
    answer: {
      type: "score",
      score,
      confidence,
      legend,
      probabilities: probabilityMap,
    },
    tokens_in: results.reduce((s, r) => s + r.tokens_in, 0),
    tokens_out: results.reduce((s, r) => s + r.tokens_out, 0),
    calls: levels.length,
  };
}

async function evaluateNoulParallel(
  client: OpenAI,
  model: string,
  state: string,
  name: string,
  q: Extract<Question, { type: "noul" | "boolean" }>,
  temperature: number
): Promise<{ answer: NoulAnswer; tokens_in: number; tokens_out: number; calls: number }> {
  let statement = q.instructions?.trim() || `The proposition "${name}" is true.`;
  if (!statement.endsWith("?") && !statement.endsWith(".")) statement += ".";

  const result = await scoreProposition(client, model, state, statement, temperature);

  return {
    answer: { type: "noul", noul: Math.round(result.probability * 1000) / 1000 },
    tokens_in: result.tokens_in,
    tokens_out: result.tokens_out,
    calls: 1,
  };
}

async function evaluateOneshot(
  client: OpenAI,
  model: string,
  state: string,
  questions: Questions,
  temperature: number
): Promise<{
  answers: Record<string, Answer>;
  tokens_in: number;
  tokens_out: number;
  calls: number;
}> {
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
    `Respond with JSON matching this shape:\n` +
    `For each choice question: { "choice": "<key>", "confidence": 0-1, "probabilities": { "<key>": 0-1, ... } }\n` +
    `For each score question: { "score": <float>, "confidence": 0-1 }\n` +
    `For each noul question: { "noul": 0-1 }\n` +
    `Top-level keys must be the question names.`;

  const completion = await client.chat.completions.create({
    model,
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    temperature,
    response_format: { type: "json_object" },
  });

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

export async function evaluate(req: EvaluateRequest): Promise<EvaluateResponse> {
  const t0 = Date.now();
  const {
    state,
    questions,
    model = "gpt-4o-mini",
    base_url,
    api_key,
    temperature = 0,
    mode = "parallel",
  } = req;

  if (!questions || Object.keys(questions).length === 0) {
    throw new Error("At least one question is required");
  }

  const stateStr =
    typeof state === "string" ? state : JSON.stringify(state, null, 2);

  const client = makeClient(api_key, base_url);

  if (mode === "parallel") {
    const entries = Object.entries(questions);

    const settled = await Promise.all(
      entries.map(async ([name, q]) => {
        if (q.type === "choice") {
          return {
            name,
            ...(await evaluateChoiceParallel(client, model, stateStr, name, q, temperature)),
          };
        }
        if (q.type === "score") {
          return {
            name,
            ...(await evaluateScoreParallel(client, model, stateStr, name, q, temperature)),
          };
        }
        return {
          name,
          ...(await evaluateNoulParallel(client, model, stateStr, name, q, temperature)),
        };
      })
    );

    const answers: Record<string, Answer> = {};
    let tokens_in = 0;
    let tokens_out = 0;
    let parallel_calls = 0;

    for (const s of settled) {
      answers[s.name] = s.answer;
      tokens_in += s.tokens_in;
      tokens_out += s.tokens_out;
      parallel_calls += s.calls;
    }

    return {
      model,
      answers,
      usage: {
        input_tokens: tokens_in || null,
        output_tokens: tokens_out || null,
      },
      meta: {
        mode: "parallel",
        latency_ms: Date.now() - t0,
        parallel_calls,
      },
    };
  }

  const oneshot = await evaluateOneshot(client, model, stateStr, questions, temperature);

  return {
    model,
    answers: oneshot.answers,
    usage: {
      input_tokens: oneshot.tokens_in || null,
      output_tokens: oneshot.tokens_out || null,
    },
    meta: {
      mode: "oneshot",
      latency_ms: Date.now() - t0,
      parallel_calls: oneshot.calls,
    },
  };
}
