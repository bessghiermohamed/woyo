# dev/ — optional developer utilities

These are **not** part of the woyo runtime. They exist to make local
development and testing possible without cloud API keys.

## zai_gateway.mjs

A tiny OpenAI-compatible HTTP gateway (ADR-3 compatibility mode) that wraps an
LLM reachable through the `z-ai-web-dev-sdk`. When the backend lacks native
tool calling, the gateway emulates it: tool schemas are serialized into the
prompt and the model's fenced-JSON reply is converted back into native
`tool_calls`.

Run it:

```bash
cd dev
bun zai_gateway.mjs          # listens on http://127.0.0.1:8787/v1
```

Then point woyo at it:

```bash
WOYO_PROVIDER=custom \
WOYO_BASE_URL=http://127.0.0.1:8787/v1 \
WOYO_API_KEY=local \
WOYO_MODEL=zai-gateway \
woyo run "your task"
```

Notes:

- Requires a working `.z-ai-config` (SDK credentials) on your machine.
- `node_modules/z-ai-web-dev-sdk` is expected to be linked/symlinked — it is
  deliberately gitignored.
- Emulated tool calling is a dev convenience. Production providers should use
  native tool calling (OpenAI, Anthropic, Groq, OpenRouter, … all support it).
