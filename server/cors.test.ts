import assert from "node:assert/strict";
import { once } from "node:events";
import {
  createServer,
  request as httpRequest,
  type IncomingHttpHeaders,
} from "node:http";
import type { AddressInfo } from "node:net";
import test from "node:test";
import cors from "cors";
import express from "express";
import { createCorsOptions, parseAllowedOrigins } from "./cors";

function appFor(raw: string | undefined) {
  const app = express();
  app.use(cors(createCorsOptions(raw)));
  app.post("/api/evaluate", (_req, res) => {
    res.json({ ok: true });
  });
  return app;
}

async function withServer(
  raw: string | undefined,
  fn: (baseUrl: string) => Promise<void>
): Promise<void> {
  const server = createServer(appFor(raw));
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address() as AddressInfo;
  try {
    await fn(`http://127.0.0.1:${address.port}`);
  } finally {
    if (server.listening) {
      const closed = once(server, "close");
      server.closeAllConnections();
      server.close();
      await closed;
    }
  }
}

function send(
  baseUrl: string,
  method: string,
  headers: Record<string, string>,
  body?: string
): Promise<{ status: number; headers: IncomingHttpHeaders; body: string }> {
  const url = new URL("/api/evaluate", baseUrl);
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname,
        method,
        headers: { ...headers, Connection: "close" },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          resolve({
            status: res.statusCode ?? 0,
            headers: res.headers,
            body: Buffer.concat(chunks).toString("utf8"),
          });
        });
      }
    );
    req.on("error", reject);
    if (body !== undefined) req.write(body);
    req.end();
  });
}

test("allowed origin receives its exact Access-Control-Allow-Origin header", async () => {
  await withServer("http://allowed.example", async (base) => {
    const res = await send(
      base,
      "POST",
      {
        Origin: "http://allowed.example",
        "Content-Type": "application/json",
      },
      "{}"
    );
    assert.equal(res.status, 200);
    assert.equal(JSON.parse(res.body).ok, true);
    assert.equal(
      res.headers["access-control-allow-origin"],
      "http://allowed.example"
    );
  });
});

test("disallowed origin receives no Access-Control-Allow-Origin header", async () => {
  await withServer("http://allowed.example", async (base) => {
    const res = await send(
      base,
      "POST",
      {
        Origin: "https://evil.example",
        "Content-Type": "application/json",
      },
      "{}"
    );
    assert.equal(res.status, 200);
    assert.equal(res.headers["access-control-allow-origin"], undefined);
  });
});

test("request without an Origin header succeeds without wildcard CORS", async () => {
  await withServer("http://allowed.example", async (base) => {
    const res = await send(
      base,
      "POST",
      { "Content-Type": "application/json" },
      "{}"
    );
    assert.equal(res.status, 200);
    assert.equal(JSON.parse(res.body).ok, true);
    assert.equal(res.headers["access-control-allow-origin"], undefined);
  });
});

test("allowed OPTIONS preflight returns CORS permission", async () => {
  await withServer("http://allowed.example", async (base) => {
    const res = await send(base, "OPTIONS", {
      Origin: "http://allowed.example",
      "Access-Control-Request-Method": "POST",
      "Access-Control-Request-Headers": "content-type",
    });
    assert.equal(res.status, 204);
    assert.equal(
      res.headers["access-control-allow-origin"],
      "http://allowed.example"
    );
    assert.ok(
      String(res.headers["access-control-allow-methods"])
        .split(",")
        .includes("POST")
    );
    assert.equal(res.headers["access-control-allow-headers"], "content-type");
  });
});

test("disallowed OPTIONS preflight returns no CORS permission", async () => {
  await withServer("http://allowed.example", async (base) => {
    const res = await send(base, "OPTIONS", {
      Origin: "https://evil.example",
      "Access-Control-Request-Method": "POST",
      "Access-Control-Request-Headers": "content-type",
    });
    assert.equal(res.headers["access-control-allow-origin"], undefined);
    assert.equal(res.headers["access-control-allow-methods"], undefined);
    assert.equal(res.headers["access-control-allow-headers"], undefined);
  });
});

test("missing allowlist denies cross-origin browsers", async () => {
  assert.deepEqual(parseAllowedOrigins(undefined), []);
  await withServer(undefined, async (base) => {
    const res = await send(
      base,
      "POST",
      {
        Origin: "https://evil.example",
        "Content-Type": "application/json",
      },
      "{}"
    );
    assert.equal(res.status, 200);
    assert.equal(res.headers["access-control-allow-origin"], undefined);
  });
});

test("comma-separated origins with whitespace are parsed exactly", async () => {
  const raw = " http://a.example , http://b.example ";
  assert.deepEqual(parseAllowedOrigins(raw), [
    "http://a.example",
    "http://b.example",
  ]);
  await withServer(raw, async (base) => {
    for (const origin of ["http://a.example", "http://b.example"]) {
      const res = await send(
        base,
        "POST",
        { Origin: origin, "Content-Type": "application/json" },
        "{}"
      );
      assert.equal(res.headers["access-control-allow-origin"], origin);
    }
  });
});

test("origin matching rejects lookalike origins", async () => {
  await withServer("http://allowed.example", async (base) => {
    for (const origin of [
      "http://allowed.example.evil",
      "https://allowed.example",
      "http://notallowed.example",
      "http://allowed.example/",
    ]) {
      const res = await send(
        base,
        "POST",
        { Origin: origin, "Content-Type": "application/json" },
        "{}"
      );
      assert.equal(res.headers["access-control-allow-origin"], undefined, origin);
    }
  });
});

test("a configured origin does not act as a wildcard", async () => {
  await withServer("https://app.example", async (base) => {
    const res = await send(
      base,
      "POST",
      {
        Origin: "https://other.example",
        "Content-Type": "application/json",
      },
      "{}"
    );
    assert.equal(res.headers["access-control-allow-origin"], undefined);
  });

  assert.deepEqual(parseAllowedOrigins("*"), ["*"]);
  await withServer("*", async (base) => {
    const res = await send(
      base,
      "POST",
      {
        Origin: "https://evil.example",
        "Content-Type": "application/json",
      },
      "{}"
    );
    assert.notEqual(res.headers["access-control-allow-origin"], "*");
    assert.equal(res.headers["access-control-allow-origin"], undefined);
  });
});
