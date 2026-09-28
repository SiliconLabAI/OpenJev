import { z } from "zod";

/**
 * API validation schemas.
 *
 * Everything the client can influence is validated and bounded here, so the
 * evaluate engine downstream can assume well-formed input. Sizes are chosen
 * to keep a single request's worst-case token spend sane.
 */

export const MAX_QUESTIONS = 50;
export const MAX_CHOICE_OPTIONS = 26; // one letter per option: A-Z
export const MAX_SCORE_LEVELS = 10;

const choiceQuestionSchema = z.object({
  type: z.literal("choice"),
  instructions: z.string().max(2000).optional(),
  criteria: z
    .record(z.string().min(1).max(200), z.string().min(1).max(200))
    .refine(
      (c) => {
        const n = Object.keys(c).length;
        return n >= 2 && n <= MAX_CHOICE_OPTIONS;
      },
      { message: `choice questions need 2-${MAX_CHOICE_OPTIONS} options` }
    ),
});

const scoreQuestionSchema = z.object({
  type: z.literal("score"),
  instructions: z.string().max(2000).optional(),
  criteria: z.array(z.string().min(1).max(200)).min(2).max(MAX_SCORE_LEVELS),
});

const noulQuestionSchema = z.object({
  type: z.enum(["noul", "boolean"]),
  instructions: z.string().max(2000).optional(),
});

const questionSchema = z.union([
  choiceQuestionSchema,
  scoreQuestionSchema,
  noulQuestionSchema,
]);

export const evaluateRequestSchema = z
  .object({
    state: z
      .union([z.string(), z.record(z.string(), z.unknown()), z.array(z.unknown())])
      .refine(
        (v) => (typeof v === "string" ? v.trim().length > 0 : true),
        "state must not be an empty string"
      ),
    questions: z
      .record(z.string().min(1).max(80), questionSchema)
      .refine(
        (qs) => {
          const n = Object.keys(qs).length;
          return n >= 1 && n <= MAX_QUESTIONS;
        },
        { message: `1-${MAX_QUESTIONS} questions per request` }
      ),
    model: z
      .string()
      .min(1)
      .max(120)
      .regex(/^[\w./:-]+$/, "invalid model name")
      .optional(),
    base_url: z.string().url().max(500).optional(),
    api_key: z.string().min(1).max(300).optional(),
    temperature: z.number().min(0).max(2).optional(),
    mode: z.enum(["parallel", "oneshot", "decider"]).optional(),
    decider_url: z.string().url().max(500).optional(),
    calibration_temperature: z.number().min(0.1).max(10).optional(),
    readout: z.enum(["logprob", "micro"]).optional(),
  })
  .strict();

export type ValidatedEvaluateRequest = z.infer<typeof evaluateRequestSchema>;

const BLOCKED_HOST_PATTERNS = [
  /^localhost$/i,
  /^127\./,
  /^0\./,
  /^10\./,
  /^192\.168\./,
  /^172\.(1[6-9]|2\d|3[01])\./,
  /^169\.254\./,
  /^\[?::1\]?$/,
  /\.internal$/i,
  /\.local$/i,
];

export class ValidationError extends Error {
  constructor(
    message: string,
    public readonly issues: { path: string; message: string }[]
  ) {
    super(message);
    this.name = "ValidationError";
  }
}

/** Validate a parsed request body, returning a typed request or throwing ValidationError. */
export function parseEvaluateRequest(body: unknown): ValidatedEvaluateRequest {
  const parsed = evaluateRequestSchema.safeParse(body);
  if (parsed.success) return parsed.data;
  const issues = parsed.error.issues.map((i) => ({
    path: i.path.join("."),
    message: i.message,
  }));
  throw new ValidationError(
    `Invalid request: ${issues.map((i) => `${i.path || "body"}: ${i.message}`).join("; ")}`,
    issues
  );
}

/**
 * SSRF guard: reject client-supplied URLs that point at loopback, private
 * ranges, or link-local metadata endpoints. Only applied when the client
 * explicitly overrides base_url / decider_url (the server-side env defaults
 * are trusted operator configuration).
 */
export function assertPublicUrl(raw: string, field: string): void {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new ValidationError(`Invalid URL in ${field}`, [
      { path: field, message: "not a valid URL" },
    ]);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new ValidationError(`Blocked ${field} protocol: ${url.protocol}`, [
      { path: field, message: `protocol ${url.protocol} not allowed` },
    ]);
  }
  if (BLOCKED_HOST_PATTERNS.some((re) => re.test(url.hostname))) {
    throw new ValidationError(
      `Blocked ${field}: ${url.hostname} resolves to a private/loopback range`,
      [{ path: field, message: "private or loopback host not allowed" }]
    );
  }
}
