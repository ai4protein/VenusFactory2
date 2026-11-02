# Configure conversational model APIs

For a local deployment, fill in a provider's API key and exact model ID in the repository's `.env`, or in **Administration → Environment**. Restart the backend and workers together, then start a new conversation and choose that model. Existing conversations keep their original model binding.

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=your-model-id
```

`MODEL` accepts one model ID. `MODELS` optionally adds several comma-separated IDs from the same account. Labels in Chat show the model names. An API key alone does not enable an example model or change the current default.

| API | Key | Model ID | Default API base |
| --- | --- | --- | --- |
| OpenAI / ChatGPT | `OPENAI_API_KEY` | `OPENAI_MODEL` | `https://api.openai.com/v1` |
| Anthropic / Claude | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` | `https://api.anthropic.com/v1` |
| xAI / Grok | `XAI_API_KEY` | `XAI_MODEL` | `https://api.x.ai/v1` |
| Zhipu / GLM | `GLM_API_KEY` | `GLM_MODEL` | `https://open.bigmodel.cn/api/paas/v4` |
| MiniMax | `MINIMAX_API_KEY` | `MINIMAX_MODEL` | `https://api.minimax.io/v1` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` | `https://api.deepseek.com/v1` |
| Google / Gemini | `GEMINI_API_KEY` | `GEMINI_MODEL` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Alibaba / Qwen | `QWEN_API_KEY` | `QWEN_MODEL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Moonshot / Kimi | `MOONSHOT_API_KEY` | `MOONSHOT_MODEL` | `https://api.moonshot.cn/v1` |

Each prefix also accepts `_BASE_URL`, `_MODELS`, `_CONTEXT_TOKENS` and `_MAX_OUTPUT_TOKENS`. Set capacity values to the selected model's documented limits; the default configuration reserves a 131,072-token context and caps output at 4,096 tokens. For models with smaller contexts, lower the configured value. When listing several models under one provider, use limits supported by every model in that list.

Regional and subscription endpoints differ. Set the base URL for the account that issued your key; use an HTTPS API base without `/chat/completions` or `/responses`. For example, MiniMax's China account can use `https://api.minimaxi.com/v1`. API usage may be billed by the provider; consumer ChatGPT/Claude subscriptions are not API credentials.

Existing `GROK_API_KEY`, `ZHIPU_API_KEY` / `ZAI_API_KEY`, `GOOGLE_API_KEY` and `DASHSCOPE_API_KEY` are accepted as key aliases for their corresponding provider only. Use the canonical model variables in the table. Claude keys that are not workspace-scoped can also set `ANTHROPIC_WORKSPACE_ID`.

## Modes and harnesses

Agent, Scientist and Researcher use the same configured model directory. Kimi Code, DeepSeek Harness and the Claude Code bridge use Chat Completions; the Claude official API is accessed through its documented OpenAI-compatible endpoint. Codex uses Responses and is available with OpenAI and compatible OpenRouter routes. Unsupported model/harness pairs remain unavailable. Model tool-calling support is required for Agent and Scientist.

The relay applies each provider's default sampling/reasoning behavior rather than forwarding free-route-specific controls. OpenAI Chat Completions uses `max_completion_tokens`. Claude's compatibility endpoint supports text, images, tools and streaming, but does not expose the full native Claude feature set, such as prompt caching. These integrations are verified with offline transport tests; actual availability depends on the configured model and account permissions.

Keys remain in the host backend. Isolated harnesses receive a temporary local relay credential. Public model catalogs and request audit logs do not include upstream keys.

## Online deployments and existing free routes

Online mode continues to use only the enabled zero-price routes in `configs/harness/free_execution_catalog.json` (or `VENUS_EXECUTION_CATALOG`). Official API environment settings do not add paid models to online selectors or permit paid execution. DMXAPI and OpenRouter keys, default selections, quota limits and zero-cost checks are unchanged. There is no automatic fallback to an official paid API.

`model-config.yaml` remains available for legacy callers; new Web conversations use the shared execution catalog and these environment settings.

Official references: [OpenAI API](https://developers.openai.com/api/reference/overview), [Claude compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk), [xAI API](https://docs.x.ai/developers/rest-api-reference/inference), [GLM compatibility](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction), [MiniMax compatibility](https://platform.minimax.io/docs/api-reference/text-openai-api).
