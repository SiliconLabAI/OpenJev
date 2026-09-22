export type QuestionType = "choice" | "score" | "noul" | "boolean";

export interface ChoiceQuestion {
  type: "choice";
  instructions?: string;
  criteria: Record<string, string>;
}

export interface ScoreQuestion {
  type: "score";
  instructions?: string;
  criteria: string[];
}

export interface NoulQuestion {
  type: "noul" | "boolean";
  instructions?: string;
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type Questions = Record<string, Question>;

/** parallel/oneshot = LLM micro-scorers; decider = Mapika/decider HTTP */
export type BackendMode = "parallel" | "oneshot" | "decider";

export interface EvaluateRequest {
  state: string | Record<string, unknown> | unknown[];
  questions: Questions;
  model?: string;
  base_url?: string;
  api_key?: string;
  temperature?: number;
  mode?: BackendMode;
  /** Override for Mapika/decider server, e.g. http://localhost:8000 */
  decider_url?: string;
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  confidence: number;
  legend: Record<string, string>;
  probabilities?: Record<string, number>;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface EvaluateResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number | null; output_tokens: number | null };
  meta?: {
    mode: BackendMode;
    latency_ms: number;
    parallel_calls: number;
    backend?: string;
  };
  error?: string;
}

export interface QuestionDraft {
  id: string;
  name: string;
  type: QuestionType;
  instructions: string;
  options: { key: string; description: string }[];
  levels: string[];
}

export const DEFAULT_EXAMPLES: {
  id: string;
  name: string;
  description: string;
  state: string;
  questions: QuestionDraft[];
}[] = [
  {
    id: "support-routing",
    name: "Support Routing",
    description: "Route a billing complaint",
    state: JSON.stringify(
      {
        subject: "Charged twice again!!",
        message:
          "This is the SECOND month in a row I've been billed twice for the Pro plan. Fix it ASAP.",
      },
      null,
      2
    ),
    questions: [
      {
        id: "q1",
        name: "department",
        type: "choice",
        instructions: "Which team should handle this?",
        options: [
          { key: "billing", description: "Charges, refunds, invoices" },
          { key: "technical", description: "Bugs or product issues" },
          { key: "account", description: "Login / access problems" },
          { key: "other", description: "Doesn't fit above" },
        ],
        levels: [],
      },
      {
        id: "q2",
        name: "urgency",
        type: "score",
        instructions: "How urgent is this?",
        options: [],
        levels: ["Low", "Medium", "High", "Critical"],
      },
      {
        id: "q3",
        name: "angry",
        type: "noul",
        instructions: "Is the customer expressing strong frustration or anger?",
        options: [],
        levels: [],
      },
    ],
  },
  {
    id: "stripe-integration",
    name: "Stripe Integration",
    description: "Failing Stripe connect",
    state:
      "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
    questions: [
      {
        id: "q1",
        name: "department",
        type: "choice",
        instructions: "Which team should handle this",
        options: [
          { key: "billing", description: "Payment or subscription issues" },
          { key: "technical", description: "Bugs or integration problems" },
          { key: "sales", description: "Pricing or account questions" },
        ],
        levels: [],
      },
      {
        id: "q2",
        name: "frustration",
        type: "score",
        instructions: "How frustrated the customer appears",
        options: [],
        levels: [
          "Calm, just stating facts",
          "Frustrated but civil",
          "Very angry, strong language",
        ],
      },
      {
        id: "q3",
        name: "is_urgent",
        type: "noul",
        instructions: "The message conveys urgency or time-sensitivity",
        options: [],
        levels: [],
      },
    ],
  },
  {
    id: "content-mod",
    name: "Content Moderation",
    description: "Toxicity check",
    state:
      "This product is absolute garbage and the CEO should be ashamed. Refund me now or I'll post this everywhere.",
    questions: [
      {
        id: "q1",
        name: "toxic",
        type: "noul",
        instructions: "Does this message contain toxic or abusive language?",
        options: [],
        levels: [],
      },
      {
        id: "q2",
        name: "action",
        type: "choice",
        instructions: "Recommended moderation action",
        options: [
          { key: "allow", description: "Publish as-is" },
          { key: "flag", description: "Flag for human review" },
          { key: "block", description: "Block / remove" },
        ],
        levels: [],
      },
    ],
  },
];

export function draftsToQuestions(drafts: QuestionDraft[]): Questions {
  const out: Questions = {};
  for (const d of drafts) {
    if (!d.name.trim()) continue;
    if (d.type === "choice") {
      const criteria: Record<string, string> = {};
      for (const o of d.options) {
        if (o.key.trim()) criteria[o.key.trim()] = o.description || o.key;
      }
      if (Object.keys(criteria).length === 0) continue;
      out[d.name] = { type: "choice", instructions: d.instructions, criteria };
    } else if (d.type === "score") {
      if (d.levels.length === 0) continue;
      out[d.name] = { type: "score", instructions: d.instructions, criteria: d.levels };
    } else {
      out[d.name] = { type: "noul", instructions: d.instructions };
    }
  }
  return out;
}
