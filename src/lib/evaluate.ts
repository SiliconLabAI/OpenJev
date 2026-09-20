/**
 * TypeScript port of openjev.evaluate — structured decisions via any
 * OpenAI-compatible chat completions API with JSON schema / response_format.
 */
import OpenAI from "openai";
import { z } from "zod";
import type {
  Answer,
  EvaluateRequest,
  EvaluateResponse,
  Question,
  Questions,
} from "./types";

function buildZodSchema(questions: Questions) {
  const shape: Record<string, z.ZodTypeAny> = {};

  for (const [name, q] of Object.entries(questions)) {
    if (q.type === "choice") {
      const keys = Object.keys(q.criteria) as [string, ...string[]];
      shape[name] = z.object({
        choice: z.enum(keys),
        confidence: z.number().min(0).max(1),
      });
    } else if (q.type === "score") {
      const n = q.criteria.length;
      shape[name] = z.object({
        score: z.number().min(0).max(Math.max(0, n - 1)),
        confidence: z.number().min(0).max(1),
      });
    } else {
      shape[name] = z.object({
        probability: z.number().min(0).max(1),
      });
    }
  }

  return z.object(shape);
}

function buildJsonSchema(questions: Questions): Record<string, unknown> {
  const properties: Record<string, unknown> = {};
  const required: string[] = [];

  for (const [name, q] of Object.entries(questions)) {
    required.push(name);
    if (q.type === "choice") {
      properties[name] = {
        type: "object",
        properties: {
          choice: { type: "string", enum: Object.keys(q.criteria) },
          confidence: { type: "number", minimum: 0, maximum: 1 },
        },
        required: ["choice", "confidence"],
        additionalProperties: false,
      };
    } else if (q.type === "score") {
      const n = q.criteria.length;
      properties[name] = {
        type: "object",
        properties: {
          score: { type: "number", minimum: 0, maximum: Math.max(0, n - 1) },
          confidence: { type: "number", minimum: 0, maximum: 1 },
        },
        required: ["score", "confidence"],
        additionalProperties: false,
      };
    } else {
      properties[name] = {
        type: "object",
        properties: {
          probability: { type: "number", minimum: 0, maximum: 1 },
        },
        required: ["probability"],
        additionalProperties: false,
      };
    }
  }

  return {
    type: "object",
    properties,
    required,
    additionalProperties: false,
  };
}

function buildPrompt(state: string, questions: Questions): {
  system: string;
  user: string;
} {
  const questionDescriptions: string[] = [];

  for (const [name, q] of Object.entries(questions)) {
    let desc = `- ${name} (${q.type}): ${q.instructions ?? ""}`;
    if (q.type === "choice") {
      const opts = Object.entries(q.criteria)
        .map(([k, v]) => `${k}: ${v}`)
        .join(", ");
      desc += `\n  Options → ${opts}`;
    } else if (q.type === "score") {
      const levels = q.criteria
        .map((lvl, i) => `${i}=${lvl}`)
        .join(" | ");
      desc += `\n  Levels → ${levels}`;
    }
    questionDescriptions.push(desc);
  }

  const system =
    "You are a precise decision engine. " +
    "Answer every question based only on the provided state. " +
    "Return probabilities / confidence that reflect genuine uncertainty. " +
    "Do not invent information.";

  const user = `STATE:
${state}

QUESTIONS:
${questionDescriptions.join("\n")}

Respond with a JSON object that strictly matches the schema.
For choice: pick one key and give a confidence 0-1.
For score: give a float score (can be between levels) + confidence.
For noul/boolean: give probability that the statement is true (0-1).
`;

  return { system, user };
}

function normalizeAnswers(
  questions: Questions,
  raw: Record<string, Record<string, number | string>>
): Record<string, Answer> {
  const answers: Record<string, Answer> = {};

  for (const [name, q] of Object.entries(questions)) {
    const data = raw[name];
    if (!data) continue;

    if (q.type === "choice") {
      const choice = String(data.choice);
      const confidence = Number(data.confidence) || 0;
      answers[name] = {
        type: "choice",
        choice,
        confidence,
        probabilities: { [choice]: confidence },
      };
    } else if (q.type === "score") {
      const score = Number(data.score) || 0;
      const confidence = Number(data.confidence) || 0;
      const legend: Record<string, string> = {};
      q.criteria.forEach((lvl, i) => {
        legend[String(i)] = lvl;
      });
      answers[name] = {
        type: "score",
        score,
        confidence,
        legend,
      };
    } else {
      answers[name] = {
        type: "noul",
        noul: Number(data.probability) || 0,
      };
    }
  }

  return answers;
}

export async function evaluate(
  req: EvaluateRequest
): Promise<EvaluateResponse> {
  const {
    state,
    questions,
    model = "gpt-4o-mini",
    base_url,
    api_key,
    temperature = 0,
  } = req;

  if (!questions || Object.keys(questions).length === 0) {
    throw new Error("At least one question is required");
  }

  const stateStr =
    typeof state === "string" ? state : JSON.stringify(state, null, 2);

  const client = new OpenAI({
    apiKey:
      api_key ||
      process.env.OPENAI_API_KEY ||
      process.env.CEREBRAS_API_KEY ||
      process.env.PUTER_API_KEY ||
      "sk-placeholder",
    baseURL: base_url || undefined,
  });

  const { system, user } = buildPrompt(stateStr, questions);
  const jsonSchema = buildJsonSchema(questions);
  const zodSchema = buildZodSchema(questions);

  // Prefer structured outputs (OpenAI-style); fall back to json_object
  let completion: OpenAI.Chat.Completions.ChatCompletion;
  try {
    completion = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "JevResponse",
          strict: true,
          schema: jsonSchema,
        },
      },
    });
  } catch {
    // Providers that don't support json_schema yet
    completion = await client.chat.completions.create({
      model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      temperature,
      response_format: { type: "json_object" },
    });
  }

  const content = completion.choices[0]?.message?.content;
  if (!content) {
    throw new Error("Empty model response");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    throw new Error(`Model returned non-JSON: ${content.slice(0, 200)}`);
  }

  const validated = zodSchema.safeParse(parsed);
  if (!validated.success) {
    // Best-effort: still try to normalize whatever we got
    console.warn("Schema validation soft-failed:", validated.error.message);
  }

  const raw = (validated.success ? validated.data : parsed) as Record<
    string,
    Record<string, number | string>
  >;

  return {
    model,
    answers: normalizeAnswers(questions, raw),
    usage: {
      input_tokens: completion.usage?.prompt_tokens ?? null,
      output_tokens: completion.usage?.completion_tokens ?? null,
    },
  };
}
