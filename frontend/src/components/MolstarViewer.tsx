import { useEffect, useRef, useState } from "react";

interface MolstarViewerProps {
  filePath: string;
  label?: string;
}

export function MolstarViewer({ filePath, label }: MolstarViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const pluginRef = useRef<any>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [errMsg, setErrMsg] = useState("");

  useEffect(() => {
    let disposed = false;

    async function init() {
      try {
        if (!containerRef.current) return;

        const resp = await fetch(
          `/api/structure/content?path=${encodeURIComponent(filePath)}`
        );
        if (!resp.ok) {
          throw new Error(`Failed to fetch structure (${resp.status})`);
        }
        const { content, format: rawFmt } = await resp.json();
        if (disposed) return;

        const [{ PluginContext }, { DefaultPluginSpec }] = await Promise.all([
          import("molstar/lib/mol-plugin/context"),
          import("molstar/lib/mol-plugin/spec"),
        ]);
        if (disposed) return;

        const container = containerRef.current!;
        container.innerHTML = "";
        const canvas = document.createElement("canvas");
        canvas.style.cssText = "width:100%;height:100%;display:block;";
        container.appendChild(canvas);

        const plugin = new PluginContext(DefaultPluginSpec());
        await plugin.init();
        if (disposed) {
          plugin.dispose();
          return;
        }

        const ok = await plugin.initViewerAsync(canvas, container);
        if (!ok) throw new Error("WebGL initialization failed");
        await plugin.canvas3dInitialized;
        if (disposed) {
          plugin.dispose();
          return;
        }

        const fmt =
          rawFmt === "cif" || rawFmt === "mmcif" ? "mmcif" : rawFmt || "pdb";
        const data = await plugin.builders.data.rawData({
          data: content,
          label: label || filePath,
        });
        const trajectory = await plugin.builders.structure.parseTrajectory(
          data,
          fmt
        );
        await plugin.builders.structure.hierarchy.applyPreset(
          trajectory,
          "default"
        );

        pluginRef.current = plugin;
        if (!disposed) setStatus("ready");
      } catch (e: any) {
        if (!disposed) {
          setErrMsg(e?.message || "Failed to load structure");
          setStatus("error");
        }
      }
    }

    init();

    return () => {
      disposed = true;
      pluginRef.current?.dispose();
      pluginRef.current = null;
    };
  }, [filePath, label]);

  const fileName = filePath.split("/").pop() || filePath;

  return (
    <div className="molstar-wrapper">
      <div className="molstar-header">
        <span className="molstar-label" title={filePath}>
          {label || fileName}
        </span>
        <a
          className="molstar-download-btn"
          href={`/api/structure/content?path=${encodeURIComponent(filePath)}&download=1`}
          download={fileName}
          title="Download structure file"
        >
          Download
        </a>
      </div>
      <div className="molstar-viewport-wrap">
        <div ref={containerRef} className="molstar-viewport" />
        {status === "loading" && (
          <div className="molstar-overlay">Loading structure…</div>
        )}
        {status === "error" && (
          <div className="molstar-overlay molstar-overlay-err">{errMsg}</div>
        )}
      </div>
    </div>
  );
}
