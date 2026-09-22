/**
 * OpenJev evaluate backends:
 * - parallel / oneshot: OpenAI-compatible chat LLMs
 * - decider: Mapika/decider HTTP (POST /v1/systemone) — real System One weights
 */
import OpenAI from "openai";
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
  return new OpenAI({
    apiKey: key,
    baseURL:
      (baseUrl && baseUrl.trim()) ||
      process.env.OPENAI_BASE_URL ||
      process.env.OPENJEV_BASE_URL ||
      undefined,
  });
}

interface ScoreCallResult {
  probability: number;
  tokens_in: number;
  tokens_out: number;
}

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
            properties: { p: { type: "number", minimum: 0, maximum: 1 } },
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
      return scoreProposition(client, model, state, statement, temperature);
    })
  );
  const probs = scoresToDistribution(results.map((r) => r.probability));
  const probabilities: Record<string, number> = {};
  keys.forEach((k, i) => {
    probabilities[k] = Math.round(probs[i] * 1000) / 1000;
  });
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;
  const sorted = [...probs].sort((a, b) => b - a);
  const gap = (sorted[0] ?? 0) - (sorted[1] ?? 0);
  const confidence = Math.min(
    1,
    Math.max(0, Math.round((0.5 * (sorted[0] ?? 0) + 0.5 * (gap + 0.5)) * 1000) / 1000)
  );
  return {
    answer: {
      type: "choice" as const,
      choice: keys[bestIdx],
      confidence,
      probabilities,
    },
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
) {
  const levels = q.criteria;
  const instructions = q.instructions ?? `Rate "${name}"`;
  const results = await Promise.all(
    levels.map((label, i) => {
      const statement = `On the scale for "${instructions}", the most appropriate rating is level ${i}: "${label}".`;
      return scoreProposition(client, model, state, statement, temperature);
    })
  );
  const probs = scoresToDistribution(results.map((r) => r.probability));
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
      type: "score" as const,
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
) {
  let statement = q.instructions?.trim() || `The proposition "${name}" is true.`;
  if (!statement.endsWith("?") && !statement.endsWith(".")) statement += ".";
  const result = await scoreProposition(client, model, state, statement, temperature);
  return {
    answer: { type: "noul" as const, noul: Math.round(result.probability * 1000) / 1000 },
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

/**
 * Call Mapika/decider HTTP server (TypeSafe wire format).
 * https://github.com/Mapika/decider — scripts/serve.sh Mapika/decider-2b 8000
 */
async function evaluateDecider(
  state: string | Record<string, unknown> | unknown[],
  questions: Questions,
  deciderUrl: string,
  apiKey?: string
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

  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

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
    decider_url,
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

    const result = await evaluateDecider(state, questions, url, api_key);

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

  if (mode === "parallel") {
    const settled = await Promise.all(
      Object.entries(questions).map(async ([name, q]) => {
        if (q.type === "choice") {
          return {
            name,
            ...(await evaluateChoiceParallel(
              client,
              model,
              stateStr,
              name,
              q,
              temperature
            )),
          };
        }
        if (q.type === "score") {
          return {
            name,
            ...(await evaluateScoreParallel(
              client,
              model,
              stateStr,
              name,
              q,
              temperature
            )),
          };
        }
        return {
          name,
          ...(await evaluateNoulParallel(
            client,
            model,
            stateStr,
            name,
            q,
            temperature
          )),
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
        backend: "openai-compatible",
      },
    };
  }

  const oneshot = await evaluateOneshot(
    client,
    model,
    stateStr,
    questions,
    temperature
  );

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
      backend: "openai-compatible",
    },
  };
}
