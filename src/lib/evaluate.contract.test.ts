/**
 * Contract tests for the evaluate() engine against a local fake
 * OpenAI-compatible server. No network access beyond 127.0.0.1.
 *
 * The mock dispatches on the request shape:
 *   - body.logprobs === true  -> letter top-logprobs (or 500 in "logprob-down" mode)
 *   - otherwise               -> JSON micro-scorer / oneshot answer
 */
import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { evaluate } from "./evaluate";
import type { EvaluateRequest } from "./types";

type Mode = "normal" | "logprob-down" | "all-down";

const state: { mode: Mode } = { mode: "normal" };
let server: http.Server;
let baseUrl = "";

function letterLogprobResponse() {
  return {
    id: "mock-1",
    object: "chat.completion",
    created: 0,
    model: "mock-model",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "B" },
        finish_reason: "stop",
        logprobs: {
          content: [
            {
              token: "B",
              logprob: -0.2,
              top_logprobs: [
                { token: " B", logprob: -0.2 },
                { token: "A", logprob: -1.5 },
              ],
            },
          ],
        },
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 1, total_tokens: 11 },
  };
}

function jsonAnswerResponse(content: string) {
  return {
    id: "mock-2",
    object: "chat.completion",
    created: 0,
    model: "mock-model",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };
}

before(async () => {
  process.env.OPENJEV_MAX_RETRIES = "1"; // fail fast inside the engine
  server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      } catch {
        body = {};
      }
      const isLogprobCall = body.logprobs === true;
      if (state.mode === "all-down" || (state.mode === "logprob-down" && isLogprobCall)) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "mock provider down" } }));
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      if (isLogprobCall) {
        res.end(JSON.stringify(letterLogprobResponse()));
      } else {
        res.end(JSON.stringify(jsonAnswerResponse('{"p": 0.9}')));
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address() as AddressInfo;
  baseUrl = `http://127.0.0.1:${addr.port}/v1`;
});

after(async () => {
  delete process.env.OPENJEV_MAX_RETRIES;
  delete process.env.OPENJEV_CALIBRATION_T;
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

function baseRequest(): EvaluateRequest {
  return {
    state: "The customer complains about billing.",
    questions: {
      dept: {
        type: "choice",
        instructions: "Which team?",
        criteria: { a: "team a", b: "team b" },
      },
    },
    model: "mock-model",
    base_url: baseUrl,
    api_key: "test-key",
    mode: "parallel",
  };
}

describe("evaluate contract", () => {
  test("logprob readout answers from letter top-logprobs", async () => {
    state.mode = "normal";
    const res = await evaluate(baseRequest());
    assert.equal(res.meta?.readout, "logprob");
    assert.equal(res.meta?.parallel_calls, 1);
    assert.equal(res.meta?.calibration_t, 1);
    assert.equal(res.meta?.fallback_used ?? null, null);
    assert.equal(res.meta?.matched_letters, 2);
    assert.equal(res.usage.input_tokens, 10);
    const dept = res.answers.dept;
    assert.equal(dept.type, "choice");
    // B (-0.2) beats A (-1.5) after temperature scaling.
    assert.equal(dept.choice, "b");
    assert.ok(dept.probabilities.b > dept.probabilities.a);
  });

  test("degrades per question to micro-scorers when logprobs are unsupported", async () => {
    state.mode = "logprob-down";
    const res = await evaluate(baseRequest());
    assert.equal(res.meta?.readout, "micro");
    assert.ok(res.meta?.fallback_used);
    assert.equal(res.meta?.fallback_questions, 1);
    assert.equal(res.meta?.failed_questions?.length ?? 0, 0);
    const dept = res.answers.dept;
    assert.equal(dept.type, "choice");
    // Micro-scorer returns p=0.9 for "a" and 0.5 for "b" -> argmax a.
    assert.equal(dept.choice, "a");
  });

  test("throws when every readout for every question fails", async () => {
    state.mode = "all-down";
    await assert.rejects(() => evaluate(baseRequest()));
  });

  test("env OPENJEV_CALIBRATION_T flows into meta", async () => {
    state.mode = "normal";
    process.env.OPENJEV_CALIBRATION_T = "2";
    const res = await evaluate(baseRequest());
    assert.equal(res.meta?.calibration_t, 2);
    delete process.env.OPENJEV_CALIBRATION_T;
  });
});
