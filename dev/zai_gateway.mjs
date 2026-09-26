/**
 * dev/zai_gateway.mjs — OPTIONAL dev utility (not part of the woyo runtime).
 *
 * A tiny OpenAI-compatible gateway for local development and testing.
 * It exposes POST /v1/chat/completions and forwards to an LLM through the
 * z-ai-web-dev-sdk. When the backend does not support native tool calling,
 * the gateway EMULATES it (ADR-3 compatibility mode): tool schemas are
 * serialized into the prompt and the model's fenced JSON reply is converted
 * back into native tool_calls.
 *
 * Use it to point woyo at a model endpoint without any cloud key:
 *   WOYO_PROVIDER=custom \
 *   WOYO_BASE_URL=http://127.0.0.1:8787/v1 \
 *   WOYO_API_KEY=local \
 *   WOYO_MODEL=zai-gateway \
 *   woyo run "your task"
 *
 * Requires: bun (or node) and a configured .z-ai-config for the SDK.
 */

import http from "node:http";
import ZAI from "z-ai-web-dev-sdk";

const PORT = process.env.GATEWAY_PORT ? Number(process.env.GATEWAY_PORT) : 8787;
let zai = null;

async function ensureZai() {
  if (!zai) zai = await ZAI.create();
  return zai;
}

// ---- OpenAI messages -> SDK messages + tool protocol -----------------------

function serializeTools(tools) {
  if (!tools || tools.length === 0) return "";
  const defs = tools
    .map((t) => {
      const fn = t.function || t;
      return `- ${fn.name}: ${fn.description}\n  parameters JSON schema: ${JSON.stringify(fn.parameters || { type: "object", properties: {} })}`;
    })
    .join("\n");
  return `\n\nTOOL PROTOCOL\nYou can call tools. Available tools:\n${defs}\n\nTo call tools, reply with ONLY a fenced JSON block:\n\`\`\`json\n{"tool_calls": [{"name": "<tool>", "arguments": {<args>}}]}\n\`\`\`\nYou may include several calls in one block. When you have the final answer instead, reply with plain text (no fences).`;
}

function convertMessages(messages) {
  const out = [];
  for (const m of messages) {
    if (m.role === "system") {
      out.push({ role: "system", content: m.content || "" });
    } else if (m.role === "user") {
      out.push({ role: "user", content: m.content || "" });
    } else if (m.role === "assistant") {
      if (m.content) out.push({ role: "assistant", content: m.content });
      if (m.tool_calls) {
        const calls = m.tool_calls
          .map((tc) => `${tc.function.name}(${tc.function.arguments})`)
          .join("; ");
        out.push({ role: "assistant", content: `[tool calls made: ${calls}]` });
      }
    } else if (m.role === "tool") {
      out.push({
        role: "user",
        content: `TOOL RESULT (${m.name || m.tool_call_id}):\n${m.content || ""}`,
      });
    }
  }
  return out;
}

function parseToolCalls(text) {
  if (!text) return null;
  const fence = text.match(/```json\s*([\s\S]*?)```/);
  const candidates = [];
  if (fence) candidates.push(fence[1].trim());
  const brace = text.match(/\{[\s\S]*\}/);
  if (brace) candidates.push(brace[0]);
  for (const cand of candidates) {
    try {
      const obj = JSON.parse(cand);
      if (obj && Array.isArray(obj.tool_calls) && obj.tool_calls.length > 0) {
        return obj.tool_calls.map((tc, i) => ({
          id: `gw-${Date.now()}-${i}`,
          type: "function",
          function: {
            name: String(tc.name || ""),
            arguments: JSON.stringify(tc.arguments || {}),
          },
        }));
      }
    } catch {
      /* try next candidate */
    }
  }
  return null;
}

// ---- server ----------------------------------------------------------------

async function handle(req, res) {
  if (req.method !== "POST" || !req.url.includes("/chat/completions")) {
    res.writeHead(404, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "only POST /v1/chat/completions" }));
    return;
  }
  const chunks = [];
  for await (const c of req) chunks.push(c);
  let body;
  try {
    body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    res.writeHead(400, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "invalid JSON" }));
    return;
  }

  const sdkMessages = convertMessages(body.messages || []);
  const toolBlock = serializeTools(body.tools);
  if (toolBlock && sdkMessages.length > 0) {
    const last = sdkMessages[sdkMessages.length - 1];
    last.content = (last.content || "") + toolBlock;
  }

  try {
    const client = await ensureZai();
    const completion = await client.chat.completions.create({
      messages: sdkMessages,
      thinking: { type: "disabled" },
      ...(body.model && !String(body.model).startsWith("zai-gateway")
        ? { model: body.model }
        : {}),
    });
    const text = completion?.choices?.[0]?.message?.content ?? "";
    const toolCalls = parseToolCalls(text);
    const reply = {
      id: `chatcmpl-gateway-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: body.model || "zai-gateway",
      choices: [
        {
          index: 0,
          message: toolCalls
            ? { role: "assistant", content: null, tool_calls: toolCalls }
            : { role: "assistant", content: text },
          finish_reason: toolCalls ? "tool_calls" : "stop",
        },
      ],
      usage: {
        prompt_tokens: completion?.usage?.prompt_tokens ?? 0,
        completion_tokens: completion?.usage?.completion_tokens ?? 0,
        total_tokens: completion?.usage?.total_tokens ?? 0,
      },
    };
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(reply));
  } catch (err) {
    res.writeHead(502, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: String(err && err.message ? err.message : err) }));
  }
}

const server = http.createServer(handle);
server.listen(PORT, "127.0.0.1", () => {
  console.log(`zai-gateway listening on http://127.0.0.1:${PORT}/v1`);
});
