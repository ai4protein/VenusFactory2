import { useEffect, useRef } from "react";
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";
import { isStructureFile, openStructure } from "../../lib/structurePreview";
import type { WorkspaceFile } from "../../lib/workspaceApi";
import type { ChatMessages } from "./chatMessages";

export type AttachmentPanel = "sources" | "library" | null;

export function ComposerFilePicker({ panel, onChange, running, t, workspaceEnabled, addComposerFiles, onWorkspaceFiles }: {
  panel: AttachmentPanel;
  onChange: (panel: AttachmentPanel) => void;
  running: boolean;
  t: ChatMessages;
  workspaceEnabled: boolean;
  addComposerFiles: (files: File[]) => Promise<void>;
  onWorkspaceFiles: (files: WorkspaceFile[]) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileSourceMenuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (panel !== "sources") return;
    function onDocPointerDown(ev: MouseEvent) {
      const root = fileSourceMenuRef.current;
      if (!root) return;
      if (ev.target instanceof Node && !root.contains(ev.target)) {
        onChange(null);
      }
    }
    function onKeyDown(ev: KeyboardEvent) {
      if (ev.key === "Escape") onChange(null);
    }
    document.addEventListener("mousedown", onDocPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onDocPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [panel, onChange]);

  return (
    <div className="file-source-inline" ref={fileSourceMenuRef}>
      <button
        type="button"
        className={`file-upload-icon-btn${running ? " disabled" : ""}${panel === "sources" ? " is-open" : ""}`}
        title={t.uploadFiles}
        aria-label={t.fileSourceAria}
        aria-haspopup="menu"
        aria-expanded={panel === "sources"}
        disabled={running}
        onClick={() => onChange(panel === "sources" ? null : "sources")}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
        </svg>
      </button>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        onChange={(e) => {
          const picked = Array.from(e.target.files || []);
          e.target.value = "";
          void addComposerFiles(picked);
        }}
        disabled={running}
        className="file-upload-hidden" hidden tabIndex={-1}
      />
      {panel === "sources" && (
        <div className="file-source-menu" role="menu" aria-label={t.fileSourceAria}>
          <button
            type="button"
            role="menuitem"
            className="file-source-menu-item"
            onClick={() => {
              onChange(null);
              fileInputRef.current?.click();
            }}
          >
            <span className="file-source-menu-title">{t.fileSourceLocal}</span>
            <span className="file-source-menu-hint">{t.fileSourceLocalHint}</span>
          </button>
          <button
            type="button"
            role="menuitem"
            className={`file-source-menu-item${workspaceEnabled ? "" : " is-disabled"}`}
            disabled={!workspaceEnabled}
            title={workspaceEnabled ? t.fromWorkspaceHint : t.fromWorkspaceOnlineHint}
            onClick={() => {
              if (!workspaceEnabled) return;
              onChange("library");
            }}
          >
            <span className="file-source-menu-title">{t.fromWorkspace}</span>
            <span className="file-source-menu-hint">
              {workspaceEnabled ? t.fromWorkspaceHint : t.fromWorkspaceOnlineHint}
            </span>
          </button>
        </div>
      )}
      <WorkspaceFilePicker
        workspaceEnabled={workspaceEnabled}
        disabled={running}
        allowMultiple
        hideTrigger
        open={panel === "library"}
        onOpenChange={open => onChange(open ? "library" : null)}
        buttonLabel={t.fromWorkspace}
        onPick={(picked) => {
          onWorkspaceFiles(picked);
          onChange(null);
        }}
      />
    </div>
  );
}

export function AttachedFiles({ files, workspaceFiles, running, lang, workspaceSuffix, onRemoveFile, onRemoveWorkspaceFile }: {
  files: File[];
  workspaceFiles: WorkspaceFile[];
  running: boolean;
  lang: string;
  workspaceSuffix: string;
  onRemoveFile: (index: number) => void;
  onRemoveWorkspaceFile: (id: string) => void;
}) {
  const removeLabel = lang === "zh" ? "移除 " : "Remove ";
  return (
    <div className="file-preview" aria-label={lang === "zh" ? "待发送文件" : "Attached files"}>
      {files.map((file, index) => (
        <span key={`${file.name}:${file.size}:${file.lastModified}`}>
          <span title={file.name}>
            {isStructureFile(file.name) ? (
              <button type="button" className="attachment-structure-link" onClick={() => {
                openStructure({ path: file.name, name: file.name, url: URL.createObjectURL(file) });
              }}>{file.name}</button>
            ) : file.name}
          </span>
          <button type="button" disabled={running} aria-label={removeLabel + file.name} onClick={() => onRemoveFile(index)}>×</button>
        </span>
      ))}
      {workspaceFiles.map(file => (
        <span key={file.id}>
          <span title={file.display_name}>{file.display_name} {workspaceSuffix}</span>
          <button type="button" disabled={running} aria-label={removeLabel + file.display_name}
            onClick={() => onRemoveWorkspaceFile(file.id)}>×</button>
        </span>
      ))}
    </div>
  );
}
