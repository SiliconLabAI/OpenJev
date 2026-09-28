# OpenJev

Open source System One–style decision playground: **state + typed questions → structured answers**.

Three backends:

| Mode | What it is |
|------|------------|
| **parallel** | LLM micro-scorers (one tiny `{p}` call per option, then softmax) |
| **oneshot** | Single structured JSON call to any OpenAI-compatible model |
| **decider** | [Mapika/decider](https://github.com/Mapika/decider) — real System One weights (calibration-aware RL on v10) |

## Tutorial

https://youtu.be/xtXq279B4Go

[![OpenJev Tutorial](https://img.youtube.com/vi/xtXq279B4Go/maxresdefault.jpg)](https://youtu.be/xtXq279B4Go)

## Quick start (LLM backends)

```bash
cd OpenJev
npm install
cp .env.example .env   # set OPENAI_API_KEY
npm run dev
```

Open http://localhost:3001 — mode **parallel** or **oneshot**.

## Integrate Mapika/decider (RLCD / System One)

[decider](https://github.com/Mapika/decider) is an open System One model family (Qwen3.5 fine-tunes).  
`decider-2b` **v10** includes calibration-aware RL. It speaks TypeSafe’s wire format: `POST /v1/systemone`.

### 1. Serve the model (needs a CUDA GPU, ~4 GB for 2B)

```bash
pip install "git+https://github.com/Mapika/decider#egg=decider[serve]"
# or: git clone https://github.com/Mapika/decider && cd decider && pip install -e ".[serve]"

scripts/serve.sh Mapika/decider-2b 8000
```

Smoke test:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "My card was charged twice.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team?",
      "criteria": { "billing": "charges, refunds", "technical": "bugs, outages" }
    },
    "refund": { "type": "noul", "instructions": "Is a refund needed?" }
  }
}'
```

### 2. Point OpenJev at it

In the UI: set **mode = decider**, **decider url = http://localhost:8000**

Or via API:

```bash
curl -X POST http://localhost:3001/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "decider",
    "decider_url": "http://localhost:8000",
    "state": "Charged twice again!!",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {
          "billing": "Charges, refunds",
          "technical": "Bugs",
          "other": "Else"
        }
      },
      "angry": { "type": "noul", "instructions": "Strong frustration?" }
    }
  }'
```

Or in `.env`:

```env
DECIDER_BASE_URL=http://localhost:8000
```

### 3. Python (no OpenJev) — direct

```python
from decider.infer import Decider
d = Decider("Mapika/decider-2b")  # ~4 GB VRAM
print(d.system_one(
    "I was charged twice for order A-104.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {
                "billing": "Charges, refunds",
                "technical": "Bugs",
            },
        },
        "refund": {"type": "noul", "instructions": "Is a refund needed?"},
    },
))
```

### How OpenJev wires it

```
UI / curl  →  POST /api/evaluate { mode: "decider", ... }
                →  OpenJev server
                →  POST {decider_url}/v1/systemone   (TypeSafe shape)
                →  Mapika/decider GPU server
                →  typed answers + probabilities
```

Same question types as TypeSafe Jev: **choice**, **score**, **noul**.

## Golden-set eval harness (Brier/ECE regression gate)

A labeled dataset (`eval/golden/dataset.jsonl`) runs through `evaluate()` and
scores **accuracy**, **Brier** and **ECE**. A committed baseline turns this
into a CI gate: any change that regresses accuracy, calibration, or both fails
the build.

```bash
# 1. Label rows in eval/golden/dataset.jsonl
#    expectations: {"choice":"key"} | {"score":<level idx>} | {"noul":<0..1>}

# 2. Create the baseline once (against your real provider), then commit it
npm run eval:golden -- --create --model gpt-4o-mini

# 3. Every later change is gated:
npm run eval:golden            # exits 1 on regression
```

Thresholds (accuracy −2pt, Brier +0.02, ECE +0.02) live in
`REGRESSION_THRESHOLDS` in `src/lib/golden.ts`. The CI `eval-regression` job
runs the check automatically when `OPENAI_API_KEY` is set. To demo the failure
path without a real provider:

```bash
MOCK_ANSWER=flat npx tsx eval/golden/mock-server.ts &   # well-calibrated mock
npx tsx scripts/eval-golden.ts eval/golden/dataset.jsonl --create \
  --base-url http://127.0.0.1:8901/v1 --model mock --api-key mock
# ...then compare with a sharp (overconfident) mock and watch the gate fail
```

## Environment

```env
# LLM backends
OPENAI_API_KEY=
# OPENAI_BASE_URL=
# OPENJEV_MODEL=gpt-4o-mini

# decider backend
DECIDER_BASE_URL=http://localhost:8000
# DECIDER_API_KEY=local
```

## Layout

```
OpenJev/
├── server/index.ts        # Express + Vite, POST /api/evaluate
├── server/loadEnv.ts
├── src/lib/evaluate.ts    # parallel | oneshot | decider
├── src/lib/types.ts
├── src/App.tsx
└── package.json
```
