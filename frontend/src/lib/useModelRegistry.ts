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
  /** Backend may set these to gate a model that is not currently usable
   * (e.g., kimi-code with no provider configured). */
  disabled?: boolean;
  disabled_reason?: string;
};

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
      label: "Agent",
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
