# OpenJev Playground

A **TanStack-style / Vite + React** playground that mirrors [TypeSafe AI’s Jev](https://docs.typesafe.ai) `/v1/systemone` interface using any **OpenAI-compatible** structured-output LLM (OpenAI, Cerebras, Groq, Together, local vLLM, Puter, etc.).

Port of the Python `openjev.evaluate` script to TypeScript, with:

- **REST API**: `POST /api/evaluate`
- **Playground UI** similar to the TypeSafe console playground (state + typed questions → calibrated answers)

## Quick start

```bash
cd openjev-app
npm install
npm run dev
```

Open **http://localhost:3001**

In the header, set:

| Field    | Example                                      |
|----------|----------------------------------------------|
| model    | `gpt-4o-mini` / `qwen-3.8-27b` / …           |
| base url | `https://api.openai.com/v1` (or leave empty) |
| api key  | your provider key                            |

Or set env vars before starting:

```bash
export OPENAI_API_KEY=sk-...
# or CEREBRAS_API_KEY / PUTER_API_KEY
npm run dev
```

## API

```http
POST /api/evaluate
Content-Type: application/json
```

```json
{
  "state": "Customer says: charged twice again, second month in a row!",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoices, refunds",
        "technical": "Bugs, outages, login issues",
        "sales": "Pricing, upgrades",
        "other": "Anything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["Low", "Medium", "High", "Critical"]
    },
    "escalate": {
      "type": "noul",
      "instructions": "Should a human review this immediately?"
    }
  },
  "model": "gpt-4o-mini",
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "temperature": 0
}
```

Response shape (same spirit as TypeSafe):

```json
{
  "model": "gpt-4o-mini",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.91,
      "probabilities": { "billing": 0.91 }
    },
    "urgency": {
      "type": "score",
      "score": 2.3,
      "confidence": 0.8,
      "legend": { "0": "Low", "1": "Medium", "2": "High", "3": "Critical" }
    },
    "escalate": {
      "type": "noul",
      "noul": 0.72
    }
  },
  "usage": {
    "input_tokens": 412,
    "output_tokens": 48
  }
}
```

## Question types

| type     | Returns                                      |
|----------|----------------------------------------------|
| `choice` | `choice` + `confidence` + `probabilities`    |
| `score`  | float `score` (can interpolate) + `confidence` + `legend` |
| `noul` / `boolean` | `noul` ∈ [0, 1] probability             |

## Project layout

```
openjev-app/
├── server/index.ts          # Express + Vite middleware, POST /api/evaluate
├── src/
│   ├── lib/
│   │   ├── evaluate.ts      # Core port of Python openjev
│   │   └── types.ts         # Request/response + example presets
│   ├── components/
│   │   ├── QuestionEditor.tsx
│   │   └── ResultsPanel.tsx
│   ├── styles/app.css
│   ├── App.tsx              # Playground UI
│   └── main.tsx
├── index.html
├── vite.config.ts
└── package.json
```

## Production

```bash
npm run build
NODE_ENV=production npm start
```

## Notes

- This is an **approximation** of Jev. Real Jev is a dedicated System One model with calibrated probabilities and no text generation. Here we use a chat LLM with JSON schema / structured outputs, so confidence is model-reported, not true logprob calibration.
- Providers that do not support `response_format: json_schema` fall back to `json_object`; the server still validates with Zod.
- API keys entered in the UI stay in `localStorage` and are sent only with evaluate requests; the server does not persist them.
```
