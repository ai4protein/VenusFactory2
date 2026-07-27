import { useEffect, useState } from "react";

export type ModelSpec = {
  id: string;
  label: string;
  provider: string;
  base_url: string;
  api_compatible: string;
  api_key_env: string;
  supports_tool_use: boolean;
  supports_json_schema: boolean;
  supports_prompt_caching: boolean;
  max_context_tokens: number;
  max_output_tokens: number;
  requires_adapter: boolean;
  /** "graph" (legacy LangGraph pipeline) or "kimi-code" (local kimi daemon). */
  engine?: string;
  /** Optional product mode hint from registry: science_agent | science_expert. */
  chat_mode?: string;
  /** Backend may set these to gate a model that is not currently usable
   * (e.g., kimi-code with no provider configured). */
  disabled?: boolean;
  disabled_reason?: string;
};

export type ChatMode = "science_agent" | "science_expert";

export const CHAT_MODE_STORAGE_KEY = "vf.chatMode";
export const SCIENCE_AGENT_MODEL_ID = "kimi-code";
/** Online mode locks Science Expert (and any graph send) to this model. */
export const ONLINE_FIXED_EXPERT_MODEL_ID = "deepseek-v4-pro";

export function isKimiEngineModel(m: Pick<ModelSpec, "id" | "engine" | "chat_mode">): boolean {
  if (m.chat_mode === "science_agent") return true;
  if (m.chat_mode === "science_expert") return false;
  return m.engine === "kimi-code" || m.id === SCIENCE_AGENT_MODEL_ID;
}

export function isGraphEngineModel(m: Pick<ModelSpec, "id" | "engine" | "chat_mode">): boolean {
  return !isKimiEngineModel(m);
}

/** Fresh page load always defaults to Science Agent (ignore prior Expert preference). */
export function readStoredChatMode(): ChatMode {
  return "science_agent";
}

export function persistChatMode(mode: ChatMode) {
  try {
    localStorage.setItem(CHAT_MODE_STORAGE_KEY, mode);
  } catch {
    // ignore
  }
}

/** First usable graph-engine model for Science Expert mode. */
export function pickExpertModelId(
  models: ModelSpec[],
  preferred?: string | null,
  opts?: { onlineFixed?: boolean }
): string {
  if (opts?.onlineFixed) {
    return ONLINE_FIXED_EXPERT_MODEL_ID;
  }
  const graphModels = models.filter((m) => isGraphEngineModel(m) && !m.disabled);
  if (preferred) {
    const hit = graphModels.find((m) => m.id === preferred);
    if (hit) return hit.id;
  }
  const fallbacks = [ONLINE_FIXED_EXPERT_MODEL_ID, "gemini-2.5-pro"];
  for (const id of fallbacks) {
    const hit = graphModels.find((m) => m.id === id);
    if (hit) return hit.id;
  }
  if (graphModels[0]) return graphModels[0].id;
  return preferred || ONLINE_FIXED_EXPERT_MODEL_ID;
}

export function chatModeFromSnapshot(opts: {
  chat_mode?: string | null;
  engine?: string | null;
  model_name?: string | null;
  models?: ModelSpec[];
}): ChatMode {
  const mode = (opts.chat_mode || "").trim();
  if (mode === "science_agent" || mode === "science_expert") return mode;
  const engine = (opts.engine || "").trim();
  if (engine === "kimi-code") return "science_agent";
  if (engine === "graph") return "science_expert";
  const modelName = (opts.model_name || "").trim();
  if (modelName) {
    const spec = (opts.models || []).find((m) => m.id === modelName);
    if (spec) return isKimiEngineModel(spec) ? "science_agent" : "science_expert";
    if (modelName === SCIENCE_AGENT_MODEL_ID) return "science_agent";
    return "science_expert";
  }
  return readStoredChatMode();
}

export type GatewaySpec = {
  id: string;
  label: string;
  base_url: string;
  api_key_env: string;
};

export type ModelRegistryResponse = {
  default_model: string;
  models: ModelSpec[];
  gateways: GatewaySpec[];
  active_gateway: string | null;
  key_status: Record<string, boolean>;
};

// Fallback registry used when GET /api/models fails — keeps the UI usable
// rather than crashing. Should mirror the backend default as closely as possible.
export const FALLBACK_REGISTRY: ModelRegistryResponse = {
  default_model: "kimi-code",
  models: [
    {
      id: "kimi-code",
      label: "Science Agent",
      provider: "kimi",
      base_url: "",
      api_compatible: "openai",
      api_key_env: "",
      supports_tool_use: true,
      supports_json_schema: false,
      supports_prompt_caching: false,
      max_context_tokens: 200000,
      max_output_tokens: 8192,
      requires_adapter: false,
      engine: "kimi-code",
      chat_mode: "science_agent",
    },
    {
      id: "deepseek-v4-pro",
      label: "DeepSeek V4 Pro",
      provider: "deepseek",
      base_url: "https://api.deepseek.com",
      api_compatible: "openai",
      api_key_env: "DEEPSEEK_API_KEY",
      supports_tool_use: true,
      supports_json_schema: false,
      supports_prompt_caching: false,
      max_context_tokens: 128000,
      max_output_tokens: 8192,
      requires_adapter: false,
      engine: "graph",
      chat_mode: "science_expert",
    },
  ],
  gateways: [],
  active_gateway: null,
  key_status: { deepseek: false },
};

export function useModelRegistry() {
  const [data, setData] = useState<ModelRegistryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch("/api/models")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<ModelRegistryResponse>;
      })
      .then((json) => {
        if (!cancelled) {
          setData(json);
          setError(null);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          // Fall back to a minimal registry so the UI keeps working offline /
          // when the backend endpoint is unavailable.
          setData(FALLBACK_REGISTRY);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [refreshTick]);

  return {
    data,
    loading,
    error,
    refresh: () => setRefreshTick((x) => x + 1),
  };
}

export async function setProviderKey(provider: string, apiKey: string): Promise<void> {
  const res = await fetch(`/api/models/keys/${encodeURIComponent(provider)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Failed to set key: HTTP ${res.status}`);
  }
}

export async function setActiveGateway(gatewayId: string | null): Promise<void> {
  const res = await fetch("/api/models/gateway", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gateway_id: gatewayId }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Failed to set gateway: HTTP ${res.status}`);
  }
}
