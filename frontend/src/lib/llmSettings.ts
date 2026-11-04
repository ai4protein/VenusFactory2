export const LLM_PROVIDERS = [
  ["DMXAPI", "DMXAPI"], ["OPENROUTER", "OpenRouter"], ["OPENAI", "OpenAI / ChatGPT"],
  ["ANTHROPIC", "Anthropic / Claude"], ["GLM", "Zhipu / GLM"], ["XAI", "xAI / Grok"],
  ["MINIMAX", "MiniMax"], ["DEEPSEEK", "DeepSeek"], ["GEMINI", "Google / Gemini"],
  ["QWEN", "Alibaba / Qwen"], ["MOONSHOT", "Moonshot / Kimi"]
] as const;
const ALIASES: Record<string,string> = {
  GROK_API_KEY:"XAI", ZHIPU_API_KEY:"GLM", ZAI_API_KEY:"GLM",
  GOOGLE_API_KEY:"GEMINI", DASHSCOPE_API_KEY:"QWEN"
};
export function llmProvider(key: string) {
  return LLM_PROVIDERS.find(([prefix]) => key.startsWith(prefix+"_") || ALIASES[key]===prefix);
}
export function llmSettingLabel(key: string, zh: boolean) {
  const provider = llmProvider(key);
  if (!provider) return undefined;
  if (ALIASES[key]) return zh ? "备用 API 密钥" : "API key alias";
  const suffix=key.slice(provider[0].length+1);
  const labels: Record<string,[string,string]> = {
    API_KEY:["API key","API 密钥"], BASE_URL:["API base URL","API 地址"],
    MODEL:["Model ID","模型 ID"], MODELS:["Additional model IDs","其他模型 ID"],
    CONTEXT_TOKENS:["Context capacity (tokens)","上下文容量（tokens）"],
    MAX_OUTPUT_TOKENS:["Maximum output tokens","最大输出 tokens"],
    WORKSPACE_ID:["Workspace ID","工作区 ID"]
  };
  return labels[suffix]?.[zh?1:0];
}
