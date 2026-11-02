import { useEffect, useRef, useState } from "react";
import type { PluginUIContext } from "molstar/lib/mol-plugin-ui/context";
import type { Root } from "react-dom/client";
import { useLang } from "../lib/i18n";
import { structureExtension, structureFormat } from "../lib/structurePreview";
import "./MolstarViewer.css";

interface Props { filePath: string; label?: string; sourceUrl?: string; format?: string; embedded?: boolean }
export function MolstarViewer({ filePath, label, sourceUrl, format, embedded = false }: Props) {
  const { lang } = useLang(); const zh = lang === "zh";
  const containerRef = useRef<HTMLDivElement>(null);
  const pluginRef = useRef<PluginUIContext | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [tools, setTools] = useState(false);
  const [representation, setRepresentation] = useState("auto");
  const [color, setColor] = useState("default");
  const [selected, setSelected] = useState(0);
  const [working, setWorking] = useState(false);
  const [polymer, setPolymer] = useState(true);
  const filename = label || filePath.split("/").pop() || filePath;
  const labelRef = useRef(filename); labelRef.current = filename;
  const download = sourceUrl || `/api/structure/content?path=${encodeURIComponent(filePath)}&download=1`;

  useEffect(() => {
    let disposed = false;
    let plugin: PluginUIContext | undefined;
    let root: Root | undefined;
    let subscription: { unsubscribe(): void } | undefined;
    let observer: ResizeObserver | undefined;
    const abort = new AbortController();
    setStatus("loading"); setError(""); setTools(false); setSelected(0); setColor("default"); setRepresentation("auto"); setPolymer(true);
    async function init() {
      try {
        const resp = await fetch(sourceUrl || `/api/structure/content?path=${encodeURIComponent(filePath)}`, { signal: abort.signal });
        if (!resp.ok) throw new Error(resp.status === 403 || resp.status === 404 ? "File unavailable" : `HTTP ${resp.status}`);
        let rawFormat = format || structureExtension(filePath);
        let content: string | ArrayBuffer;
        if (sourceUrl) content = rawFormat === "bcif" ? await resp.arrayBuffer() : await resp.text();
        else { const body = await resp.json(); content = body.content; rawFormat = body.format || rawFormat; }
        if (disposed) return;
        const [{ createPluginUI }, { DefaultPluginUISpec }, { createRoot }, { PluginConfig }, { Color }] = await Promise.all([
          import("molstar/lib/mol-plugin-ui"), import("molstar/lib/mol-plugin-ui/spec"), import("react-dom/client"),
          import("molstar/lib/mol-plugin/config"), import("molstar/lib/mol-util/color"), import("molstar/build/viewer/molstar.css"), import("molstar/build/viewer/theme/light.css"),
        ]);
        if (disposed || !containerRef.current) return;
        const spec = DefaultPluginUISpec();
        plugin = await createPluginUI({ target: containerRef.current,
          render: (element, target) => { root = createRoot(target); root.render(element); },
          spec: { ...spec,
            layout: { initial: { isExpanded: false, showControls: true, controlsDisplay: "portrait", regionState: { top: "full", left: "hidden", right: "hidden", bottom: "hidden" } } },
            components: { ...spec.components, controls: { left: "none", bottom: "none" }, remoteState: "none", disableDragOverlay: true },
            config: [...(spec.config || []), [PluginConfig.Viewport.ShowExpand, false], [PluginConfig.Viewport.ShowControls, false],
              [PluginConfig.Viewport.ShowSelectionMode, true], [PluginConfig.Viewport.ShowScreenshotControls, true],
              [PluginConfig.VolumeStreaming.Enabled, false]],
          },
        });
        if (disposed) { root?.unmount(); plugin.dispose(); return; }
        if (!plugin.canvas3d) throw new Error("WebGL unavailable");
        plugin.canvas3d.setProps({ renderer: { backgroundColor: Color(0xffffff), selectColor: Color(0xf2b544), highlightColor: Color(0x76b8dc) } });
        const data = await plugin.builders.data.rawData({ data: content, label: labelRef.current });
        const trajectory = await plugin.builders.structure.parseTrajectory(data, structureFormat(rawFormat) as "pdb" | "mmcif" | "mol" | "sdf" | "mol2");
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, "default");
        if (disposed) return;
        const structures = plugin.managers.structure.hierarchy.current.structures;
        const hasPolymer = structures.some(s => (s.cell.obj?.data.polymerResidueCount || 0) > 0);
        setPolymer(hasPolymer);
        if (!hasPolymer) {
          const { PluginCommands } = await import("molstar/lib/mol-plugin/commands");
          if (disposed) return;
          await PluginCommands.Layout.Update(plugin, { state: { regionState: { ...plugin.layout.state.regionState, top: "hidden" } } });
          plugin.managers.interactivity.setProps({ granularity: "element" });
          const sphere = structures[0]?.cell.obj?.data.boundary.sphere;
          if (sphere) plugin.managers.camera.focusSphere(sphere, { minRadius: 2, extraRadius: 2 });
        }
        plugin.selectionMode = true;
        pluginRef.current = plugin;
        subscription = plugin.managers.structure.selection.events.changed.subscribe(() => setSelected(plugin!.managers.structure.selection.elementCount()));
        observer = new ResizeObserver(() => plugin?.canvas3d?.requestResize());
        if (containerRef.current) observer.observe(containerRef.current);
        setStatus("ready");
      } catch (e) {
        if (!disposed) { setError(e instanceof Error ? e.message : "Could not load structure"); setStatus("error"); }
      }
    }
    void init();
    return () => { disposed = true; abort.abort(); observer?.disconnect(); subscription?.unsubscribe(); pluginRef.current = null;
      const mountedRoot = root; const mountedPlugin = plugin;
      // React owns this nested UI root; unmount it outside the parent commit.
      queueMicrotask(() => { mountedRoot?.unmount(); mountedPlugin?.dispose(); });
    };
  }, [filePath, sourceUrl, format, revision]);

  async function run(action: (plugin: PluginUIContext) => Promise<unknown> | void) {
    const p = pluginRef.current; if (!p || working) return;
    setWorking(true); setError("");
    try { await action(p); } catch { setError(zh ? "操作未完成，请重试" : "Could not apply this change. Try again."); }
    finally { setWorking(false); }
  }
  async function changeRepresentation(value: string) {
    await run(async p => {
      for (const s of p.managers.structure.hierarchy.current.structures) {
        if (value === "auto") await p.builders.structure.representation.applyPreset(s.cell, "auto");
        else for (const c of s.components) {
          await p.managers.structure.component.removeRepresentations([c]);
          await p.managers.structure.component.addRepresentation([c], value === "cartoon" && !c.cell.obj?.data.polymerResidueCount ? "ball-and-stick" : value);
        }
      }
      setRepresentation(value); setColor("default");
    });
  }
  const ready = status === "ready" && !working;
  return <div className={`molstar-workbench${embedded ? " is-embedded" : ""}`}>
    {!embedded && <header className="structure-viewer-title"><strong>{filename}</strong><a href={download} download={filename}>{zh ? "下载" : "Download"}</a></header>}
    <div className="structure-quick-tools" aria-label={zh ? "结构操作" : "Structure controls"}>
      <select aria-label={zh ? "结构显示" : "Representation"} disabled={!ready} value={representation} onChange={e => void changeRepresentation(e.target.value)}>
        <option value="auto">{zh ? "自动显示" : "Automatic"}</option>{polymer && <option value="cartoon">{zh ? "卡通" : "Cartoon"}</option>}<option value="ball-and-stick">{zh ? "球棍" : "Ball & stick"}</option><option value="spacefill">{zh ? "空间填充" : "Spacefill"}</option><option value="molecular-surface">{zh ? "分子表面" : "Surface"}</option>
      </select>
      <select aria-label={zh ? "结构着色" : "Color by"} disabled={!ready} value={color} onChange={e => { const value = e.target.value; void run(async p => { for (const s of p.managers.structure.hierarchy.current.structures) await p.managers.structure.component.updateRepresentationsTheme(s.components, { color: value as any }); setColor(value); }); }}>
        {[["default", "默认着色", "Default color"], ["chain-id", "按链", "Chain"], ["element-symbol", "按元素", "Element"], ["residue-name", "按残基", "Residue"], ["secondary-structure", "二级结构", "Secondary structure"], ["sequence-id", "序列位置", "Sequence position"], ["hydrophobicity", "疏水性", "Hydrophobicity"]].map(([id, cn, en]) => <option key={id} value={id}>{zh ? cn : en}</option>)}
      </select>
      <button disabled={!ready} onClick={() => { const p = pluginRef.current; const sphere = p?.managers.structure.hierarchy.current.structures[0]?.cell.obj?.data.boundary.sphere; if (!polymer && sphere) p?.managers.camera.focusSphere(sphere, { minRadius: 2, extraRadius: 2 }); else p?.managers.camera.reset(); }} title={zh ? "重置视角" : "Reset view"}>{zh ? "复位" : "Reset"}</button>
      <button disabled={!ready || !selected} onClick={() => { const p = pluginRef.current; if (p) p.managers.camera.focusSphere(p.managers.structure.selection.getBoundary().sphere); }}>{zh ? "聚焦选择" : "Focus"}</button>
      <button disabled={!ready || !selected} onClick={() => pluginRef.current?.managers.interactivity.lociSelects.deselectAll()}>{zh ? "清除选择" : "Clear"}</button>
      <button disabled={!ready} aria-pressed={tools} onClick={() => void run(async p => { const { PluginCommands } = await import("molstar/lib/mol-plugin/commands"); await PluginCommands.Layout.Update(p, { state: { regionState: { ...p.layout.state.regionState, right: tools ? "hidden" : "full" } } }); setTools(!tools); })}>{zh ? "更多工具" : "Tools"}</button>
    </div>
    <div className="structure-canvas-wrap"><div ref={containerRef} className="structure-molstar-host"/>
      {status === "loading" && <div className="structure-loading" role="status">{zh ? "正在加载结构…" : "Loading structure…"}</div>}
      {status === "error" && <div className="structure-loading" role="alert"><span>{zh ? "结构未能加载" : "Could not load structure"}</span><small>{error}</small><button onClick={() => setRevision(n => n + 1)}>{zh ? "重试" : "Retry"}</button><a href={download} download={filename}>{zh ? "下载文件" : "Download file"}</a></div>}
    </div>
    <footer className="structure-interaction-hint"><span>{polymer ? (zh ? "拖选序列以选择残基 · 拖动旋转 · 滚轮缩放" : "Drag across the sequence to select · Drag to rotate · Scroll to zoom") : (zh ? "点击选择原子 · 拖动旋转 · 滚轮缩放" : "Click to select atoms · Drag to rotate · Scroll to zoom")}</span>{selected > 0 && <strong>{selected} {zh ? "个原子已选" : "atoms selected"}</strong>}</footer>
    {status === "ready" && error && <div className="structure-action-error" role="alert">{error}</div>}
  </div>;
}
