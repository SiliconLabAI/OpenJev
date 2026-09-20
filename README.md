# OpenJev

Open approximation of [TypeSafe Jev](https://docs.typesafe.ai) — a **System One** style decision engine.

Instead of one flaky “return a giant JSON blob” call, OpenJev uses a **parallel sampler**:

1. **Fixed answer space** — no free-form text generation
2. **Each option scored independently** against the same state
3. **Scores normalized** (logit → softmax) into a probability distribution
4. **All questions run in parallel** (`Promise.all`)

That mirrors how Jev is described: parallel evaluation over a declared answer set, not sequential token generation.

## Quick start

```bash
cd OpenJev   # folder name; package is "OpenJev"
npm install
npm run dev
```

Open **http://localhost:3001**

| Field    | Example                            |
|----------|------------------------------------|
| mode     | `parallel` (default) or `oneshot`  |
| model    | `gpt-4o-mini`, `qwen-3.8-27b`, …   |
| base url | provider base, or empty for OpenAI |
| api key  | your key                           |

```bash
export OPENAI_API_KEY=sk-...
npm run dev
```

## Why parallel mode is more reliable

| Mode | Behavior | Failure mode |
|------|----------|--------------|
| **parallel** (default) | One tiny `{"p": 0–1}` call **per option**, then softmax | Rare — each call is tiny and constrained |
| **oneshot** | One big structured JSON for all questions | Model drops keys, invents labels, invalid JSON |

Example: choice with 4 options → 4 parallel micro-calls. Score with 4 levels → same. Noul → 1 call. Questions themselves also run in parallel.

## API

```http
POST /api/evaluate
Content-Type: application/json
```

```json
{
  "state": "Charged twice again!! Second month in a row.",
  "mode": "parallel",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges, refunds, invoices",
        "technical": "Bugs or product issues",
        "other": "Doesn't fit"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent?",
      "criteria": ["Low", "Medium", "High", "Critical"]
    },
    "angry": {
      "type": "noul",
      "instructions": "Strong frustration or anger?"
    }
  },
  "model": "gpt-4o-mini"
}
```

Response includes full distributions + meta:

```json
{
  "model": "gpt-4o-mini",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.82,
      "probabilities": { "billing": 0.71, "technical": 0.12, "other": 0.17 }
    },
    "urgency": {
      "type": "score",
      "score": 2.4,
      "confidence": 0.75,
      "legend": { "0": "Low", "1": "Medium", "2": "High", "3": "Critical" },
      "probabilities": { "0": 0.05, "1": 0.15, "2": 0.45, "3": 0.35 }
    },
    "angry": { "type": "noul", "noul": 0.88 }
  },
  "usage": { "input_tokens": 1840, "output_tokens": 96 },
  "meta": { "mode": "parallel", "latency_ms": 620, "parallel_calls": 9 }
}
```

## How scoring works

For each candidate (choice key or score level):

```
STATEMENT = "The correct answer is <key> (<description>)."
→ model returns { "p": 0.0 … 1.0 }
```

Independent `p` values → logits via `logit(p) = log(p/(1-p))` → **softmax** → distribution.

- **choice** → argmax + confidence from top-1 / gap
- **score** → expected value of the discrete distribution (interpolation allowed)
- **noul** → single `p`

## Not real Jev

OpenJev uses ordinary chat LLMs as micro-scorers. Real Jev is a specialized System One model (RLCD, custom parallel sampler, ~70–500 ms). This is an **open architectural approximation** of the contract, not a weight-compatible reimplementation.
