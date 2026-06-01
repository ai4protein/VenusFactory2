import { useEffect, useMemo, useState } from "react";
import { PageFooter } from "../components/PageFooter";
import {
  getInsightsIpDistribution,
  getInsightsMap,
  getInsightsOverview,
  getInsightsTokenUsage,
  getInsightsToolCalls,
  type InsightsIpDistribution,
  type InsightsMap,
  type InsightsOverview,
  type InsightsTokenUsage,
  type InsightsToolCalls
} from "../lib/api";
import { useLang } from "../lib/i18n";
import { useDocumentMeta } from "../lib/useDocumentMeta";

const STRINGS = {
  en: {
    title: "Insights",
    subtitle: "Operational analytics for tools, traffic, and token consumption.",
    timeRange: "Time Range",
    refresh: "Refresh",
    loadingDash: "Loading insights dashboard...",
    kpiTotalCalls: "Total Tool Calls",
    kpiSuccessRate: "Success rate",
    kpiActiveOwners: "Active Owners",
    kpiUniqueIps: "Unique IPs",
    kpiTotalTokens: "Total Tokens",
    kpiTokensIn: "In",
    kpiTokensOut: "Out",
    kpiEstCost: "Estimated Cost",
    kpiCostNote: "Based on configured model prices",
    panelTrend: "Tool Call Trend",
    panelIp: "IP Distribution",
    panelIpUnit: "country",
    panelMap: "Access Map",
    panelMapUnit: "offline geoip",
    panelToken: "Token Consumption",
    panelTokenUnit: "daily rollup",
    colBucket: "Bucket",
    colTotalTokens: "Total Tokens",
    colEstCost: "Estimated Cost",
    noToolCalls: "No tool calls yet.",
    noAccess: "No access events in range.",
    noTokens: "No token usage in range.",
    noData: "No data",
    mapAriaLabel: "IP access world map",
    errOverview: "Failed to load insights overview.",
    errInsights: "Failed to load insights."
  },
  zh: {
    title: "运行洞察",
    subtitle: "工具、流量与 Token 消耗的运营分析。",
    timeRange: "时间范围",
    refresh: "刷新",
    loadingDash: "正在加载洞察仪表盘…",
    kpiTotalCalls: "工具调用总数",
    kpiSuccessRate: "成功率",
    kpiActiveOwners: "活跃用户",
    kpiUniqueIps: "独立 IP",
    kpiTotalTokens: "Token 总量",
    kpiTokensIn: "输入",
    kpiTokensOut: "输出",
    kpiEstCost: "预估费用",
    kpiCostNote: "基于已配置的模型价格",
    panelTrend: "工具调用趋势",
    panelIp: "IP 分布",
    panelIpUnit: "按国家",
    panelMap: "访问地图",
    panelMapUnit: "离线 geoip",
    panelToken: "Token 消耗",
    panelTokenUnit: "按日汇总",
    colBucket: "分桶",
    colTotalTokens: "Token 总量",
    colEstCost: "预估费用",
    noToolCalls: "暂无工具调用。",
    noAccess: "此时间段无访问记录。",
    noTokens: "此时间段无 Token 消耗。",
    noData: "无数据",
    mapAriaLabel: "IP 访问世界地图",
    errOverview: "加载洞察概览失败。",
    errInsights: "加载洞察数据失败。"
  }
};

type TimeWindow = "24h" | "7d" | "30d";

function buildRange(window: TimeWindow) {
  const to = new Date();
  const from = new Date(to.getTime());
  if (window === "24h") {
    from.setHours(from.getHours() - 24);
  } else if (window === "7d") {
    from.setDate(from.getDate() - 7);
  } else {
    from.setDate(from.getDate() - 30);
  }
  return { fromIso: from.toISOString(), toIso: to.toISOString() };
}

function formatNumber(value: number) {
  return new Intl.NumberFormat().format(value);
}

function formatUsd(value: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value);
}

function Sparkline({ values, noDataLabel }: { values: number[]; noDataLabel: string }) {
  if (!values.length) return <div className="insights-empty-inline">{noDataLabel}</div>;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const span = Math.max(max - min, 1);
  const points = values
    .map((v, i) => {
      const x = (i / Math.max(values.length - 1, 1)) * 100;
      const y = 100 - ((v - min) / span) * 100;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="insights-sparkline">
      <polyline points={points} />
    </svg>
  );
}

const COUNTRY_CENTROIDS: Record<string, { lat: number; lon: number }> = {
  US: { lat: 39, lon: -98 },
  CN: { lat: 35, lon: 103 },
  IN: { lat: 22, lon: 79 },
  DE: { lat: 51, lon: 10 },
  FR: { lat: 46, lon: 2 },
  GB: { lat: 55, lon: -3 },
  JP: { lat: 36, lon: 138 },
  KR: { lat: 36, lon: 128 },
  SG: { lat: 1.3, lon: 103.8 },
  AU: { lat: -25, lon: 133 },
  CA: { lat: 56, lon: -106 },
  BR: { lat: -10, lon: -55 },
  ZA: { lat: -30, lon: 25 },
  RU: { lat: 60, lon: 100 },
  ES: { lat: 40, lon: -4 },
  IT: { lat: 42.5, lon: 12.5 },
  NL: { lat: 52.3, lon: 5.3 },
  SE: { lat: 60, lon: 18 },
  CH: { lat: 46.8, lon: 8.2 },
  UNKNOWN: { lat: 0, lon: 0 }
};

function toMapXY(lat: number, lon: number) {
  const x = ((lon + 180) / 360) * 1000;
  const y = ((90 - lat) / 180) * 500;
  return { x, y };
}

function WorldMapPanel({ mapData, mapAriaLabel }: { mapData: InsightsMap["data"]["rows"]; mapAriaLabel: string }) {
  const maxIntensity = Math.max(...mapData.map((row) => row.intensity), 1);
  return (
    <svg viewBox="0 0 1000 500" className="insights-world-map" role="img" aria-label={mapAriaLabel}>
      <rect x="0" y="0" width="1000" height="500" rx="14" className="insights-world-map-bg" />
      {[125, 250, 375].map((y) => (
        <line key={`lat-${y}`} x1="0" x2="1000" y1={y} y2={y} className="insights-world-map-grid" />
      ))}
      {[250, 500, 750].map((x) => (
        <line key={`lon-${x}`} x1={x} x2={x} y1="0" y2="500" className="insights-world-map-grid" />
      ))}
      {mapData.map((row) => {
        const centroid = COUNTRY_CENTROIDS[row.country_code] || COUNTRY_CENTROIDS.UNKNOWN;
        const { x, y } = toMapXY(centroid.lat, centroid.lon);
        const ratio = row.intensity / maxIntensity;
        return (
          <g key={row.country_code}>
            <circle cx={x} cy={y} r={4 + ratio * 14} className="insights-world-map-dot" />
            <text x={x} y={y - 8} className="insights-world-map-label">
              {row.country_code}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function SettingsInsightsPage() {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
  const [window, setWindow] = useState<TimeWindow>("7d");
  const [reloadSeed, setReloadSeed] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [overview, setOverview] = useState<InsightsOverview | null>(null);
  const [toolCalls, setToolCalls] = useState<InsightsToolCalls | null>(null);
  const [ipDist, setIpDist] = useState<InsightsIpDistribution | null>(null);
  const [tokenUsage, setTokenUsage] = useState<InsightsTokenUsage | null>(null);
  const [mapData, setMapData] = useState<InsightsMap | null>(null);

  useEffect(() => {
    let alive = true;
    async function loadAll() {
      setLoading(true);
      setError("");
      try {
        const { fromIso, toIso } = buildRange(window);
        const [overviewResult, toolCallsResult, ipResult, tokenResult, mapResult] = await Promise.allSettled([
          getInsightsOverview(fromIso, toIso),
          getInsightsToolCalls(fromIso, toIso, "day"),
          getInsightsIpDistribution(fromIso, toIso, "country"),
          getInsightsTokenUsage(fromIso, toIso, "day"),
          getInsightsMap(fromIso, toIso)
        ]);
        if (!alive) return;

        if (overviewResult.status !== "fulfilled") {
          const reason = overviewResult.reason;
          throw reason instanceof Error ? reason : new Error(t.errOverview);
        }
        setOverview(overviewResult.value);

        setToolCalls(toolCallsResult.status === "fulfilled" ? toolCallsResult.value : null);
        setIpDist(ipResult.status === "fulfilled" ? ipResult.value : null);
        setTokenUsage(tokenResult.status === "fulfilled" ? tokenResult.value : null);
        setMapData(mapResult.status === "fulfilled" ? mapResult.value : null);
      } catch (err) {
        if (!alive) return;
        setError(err instanceof Error ? err.message : t.errInsights);
      } finally {
        if (alive) setLoading(false);
      }
    }
    void loadAll();
    return () => {
      alive = false;
    };
  }, [window, reloadSeed]);

  const trendValues = useMemo(
    () => (toolCalls?.data.rows || []).map((row) => row.calls),
    [toolCalls]
  );

  const topTools = useMemo(() => {
    if (!toolCalls) return [];
    const bucket = new Map<string, number>();
    for (const row of toolCalls.data.rows) {
      const key = row.tool_name || row.bucket;
      bucket.set(key, (bucket.get(key) || 0) + row.calls);
    }
    return [...bucket.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  }, [toolCalls]);

  return (
    <div className="settings-page insights-page">
      <header className="chat-header">
        <div>
          <h2>{t.title}</h2>
          <p>{t.subtitle}</p>
        </div>
      </header>

      <section className="insights-filter-bar chat-panel">
        <div className="insights-filter-group">
          <span>{t.timeRange}</span>
          <div className="insights-segment">
            {(["24h", "7d", "30d"] as TimeWindow[]).map((item) => (
              <button
                key={item}
                type="button"
                className={item === window ? "active" : ""}
                onClick={() => setWindow(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
        <button type="button" onClick={() => setReloadSeed((prev) => prev + 1)}>
          {t.refresh}
        </button>
      </section>

      {loading && <div className="chat-panel chat-empty">{t.loadingDash}</div>}
      {!loading && error && <div className="error-box">{error}</div>}

      {!loading && !error && overview && (
        <section className="insights-grid">
          <article className="insights-kpi-card">
            <div className="insights-kpi-title">{t.kpiTotalCalls}</div>
            <div className="insights-kpi-value">{formatNumber(overview.data.total_calls)}</div>
            <div className="insights-kpi-meta">{t.kpiSuccessRate} {overview.data.success_rate.toFixed(1)}%</div>
          </article>
          <article className="insights-kpi-card">
            <div className="insights-kpi-title">{t.kpiActiveOwners}</div>
            <div className="insights-kpi-value">{formatNumber(overview.data.active_owners)}</div>
            <div className="insights-kpi-meta">{t.kpiUniqueIps} {formatNumber(overview.data.unique_ips)}</div>
          </article>
          <article className="insights-kpi-card">
            <div className="insights-kpi-title">{t.kpiTotalTokens}</div>
            <div className="insights-kpi-value">{formatNumber(overview.data.total_tokens)}</div>
            <div className="insights-kpi-meta">
              {t.kpiTokensIn} {formatNumber(overview.data.input_tokens)} / {t.kpiTokensOut} {formatNumber(overview.data.output_tokens)}
            </div>
          </article>
          <article className="insights-kpi-card">
            <div className="insights-kpi-title">{t.kpiEstCost}</div>
            <div className="insights-kpi-value">{formatUsd(overview.data.estimated_cost_usd)}</div>
            <div className="insights-kpi-meta">{t.kpiCostNote}</div>
          </article>
        </section>
      )}

      {!loading && !error && (
        <section className="insights-board-grid">
          <article className="chat-panel insights-panel">
            <div className="insights-panel-head">
              <h3>{t.panelTrend}</h3>
              <span>{toolCalls?.data.group_by || "day"}</span>
            </div>
            <Sparkline values={trendValues} noDataLabel={t.noData} />
            <div className="insights-top-list">
              {topTools.map(([tool, count]) => (
                <div className="insights-top-item" key={tool}>
                  <span>{tool}</span>
                  <strong>{formatNumber(count)}</strong>
                </div>
              ))}
              {topTools.length === 0 && <div className="chat-empty">{t.noToolCalls}</div>}
            </div>
          </article>

          <article className="chat-panel insights-panel">
            <div className="insights-panel-head">
              <h3>{t.panelIp}</h3>
              <span>{t.panelIpUnit}</span>
            </div>
            <div className="insights-country-list">
              {(ipDist?.data.rows || []).slice(0, 8).map((row) => (
                <div className="insights-country-item" key={`${row.country_code}-${row.region}`}>
                  <span>{row.country_code}</span>
                  <span>PV {formatNumber(row.pv)} / UV {formatNumber(row.uv)}</span>
                </div>
              ))}
              {(ipDist?.data.rows || []).length === 0 && <div className="chat-empty">{t.noAccess}</div>}
            </div>
          </article>

          <article className="chat-panel insights-panel insights-panel-map">
            <div className="insights-panel-head">
              <h3>{t.panelMap}</h3>
              <span>{t.panelMapUnit}</span>
            </div>
            <WorldMapPanel mapData={mapData?.data.rows || []} mapAriaLabel={t.mapAriaLabel} />
          </article>

          <article className="chat-panel insights-panel insights-panel-token">
            <div className="insights-panel-head">
              <h3>{t.panelToken}</h3>
              <span>{t.panelTokenUnit}</span>
            </div>
            <div className="insights-token-table">
              <div className="insights-token-row insights-token-head">
                <span>{t.colBucket}</span>
                <span>{t.colTotalTokens}</span>
                <span>{t.colEstCost}</span>
              </div>
              {(tokenUsage?.data.rows || []).slice(0, 10).map((row) => (
                <div className="insights-token-row" key={`${row.bucket}-${row.model}-${row.tool_name}`}>
                  <span>{row.bucket}</span>
                  <span>{formatNumber(row.total_tokens)}</span>
                  <span>{formatUsd(row.estimated_cost_usd)}</span>
                </div>
              ))}
              {(tokenUsage?.data.rows || []).length === 0 && <div className="chat-empty">{t.noTokens}</div>}
            </div>
          </article>
        </section>
      )}
      <PageFooter />
    </div>
  );
}
