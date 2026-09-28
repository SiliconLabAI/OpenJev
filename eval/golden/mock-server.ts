/**
 * Mock OpenAI-compatible server for exercising the eval harness without a
 * real provider. Answers deterministic letter logprobs derived from the
 * OPTIONS list in the prompt.
 *
 *   MOCK_ANSWER=first  npx tsx eval/golden/mock-server.ts   # sharp, first option wins
 *   MOCK_ANSWER=last   npx tsx eval/golden/mock-server.ts   # sharp, last option wins
 *   MOCK_ANSWER=flat   npx tsx eval/golden/mock-server.ts   # max-entropy answers
 *
 * "flat" reliably trips the Brier/ECE regression gates (confident baseline
 * vs uniform answers), which is what you want when demoing the CI failure
 * path.
 */
import http from "node:http";

const MODE = (process.env.MOCK_ANSWER ?? "first") as "first" | "last" | "flat";
const PORT = Number(process.env.MOCK_PORT) || 8901;

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

interface ChatBody {
  logprobs?: boolean;
  messages?: { role: string; content: string }[];
  response_format?: { type: string };
}

function extractOptions(userContent: string): string[] {
  // Lines shaped "A. key — description" from buildOptionLetterPrompt.
  const lines = userContent.split("\n").filter((l) => /^[A-Z]\. /.test(l));
  return lines.map((l) => l.slice(3).split(" — ")[0].trim());
}

function letterResponse(body: ChatBody) {
  const user = body.messages?.find((m) => m.role === "user")?.content ?? "";
  const options = extractOptions(user);
  const count = Math.max(1, options.length);
  const winner = MODE === "last" ? count - 1 : 0;

  const top = LETTERS.slice(0, count).map((letter, i) => ({
    token: i === winner && MODE !== "flat" ? letter : ` ${letter}`,
    logprob: MODE === "flat" ? -1.0 : i === winner ? -0.1 : -2.0 - i,
  }));

  return {
    id: "mock-letter",
    object: "chat.completion",
    created: 0,
    model: "mock",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: LETTERS[winner] },
        finish_reason: "stop",
        logprobs: { content: [{ token: LETTERS[winner], logprob: -0.1, top_logprobs: top }] },
      },
    ],
    usage: { prompt_tokens: 20, completion_tokens: 1, total_tokens: 21 },
  };
}

function jsonProbResponse() {
  const p = MODE === "flat" ? 0.5 : MODE === "last" ? 0.4 : 0.9;
  return {
    id: "mock-json",
    object: "chat.completion",
    created: 0,
    model: "mock",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: JSON.stringify({ p }) },
        finish_reason: "stop",
      },
    ],
    usage: { prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 },
  };
}

http
  .createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      let body: ChatBody = {};
      try {
        body = JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
      } catch {
        body = {};
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify(body.logprobs === true ? letterResponse(body) : jsonProbResponse())
      );
    });
  })
  .listen(PORT, () => {
    console.log(`[mock-openai] listening on http://127.0.0.1:${PORT}/v1 (mode=${MODE})`);
  });
