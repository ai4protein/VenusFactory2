import { useFileTransfer } from "../lib/useFileTransfer";
import { mergeFiles } from "../lib/fileTransfer";
import { validateUploads } from "../lib/uploadLimits";
import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useLang } from "../lib/i18n";
import { uploadResearchInputs, type ResearchAttachment } from "../lib/scienceGoals";

export function ResearchInputs({ ensureSession, value, onChange, onBusy, buttonTarget, transferTarget, disabled = false }: { ensureSession: () => Promise<string>; value: ResearchAttachment[]; onChange: (files: ResearchAttachment[]) => void; onBusy?: (busy: boolean) => void; buttonTarget?: HTMLElement | null; transferTarget?: HTMLElement | null; disabled?: boolean }) {
  const { lang } = useLang(); const zh = lang === "zh";
  const input = useRef<HTMLInputElement>(null);
  const [root, setRoot] = useState<HTMLDivElement | null>(null);
  const lock = useRef(false); const currentValue = useRef(value); currentValue.current = value;
  const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  async function upload(files: File[]) {
    if (!files.length) return;
    if (lock.current || disabled) { setError(zh ? "请等待当前操作完成后添加文件。" : "Wait for the current operation before attaching files."); return; }
    files = mergeFiles([], files);
    if (value.length + files.length > 32) { setError(zh ? "最多上传 32 个文件" : "Select up to 32 files"); return; }
    lock.current = true; setBusy(true); onBusy?.(true); setError("");
    try { await validateUploads(files); const sid = await ensureSession(); const result = await uploadResearchInputs(sid, files); onChange([...currentValue.current, ...result.files]); }
    catch (e) { setError((e as Error).message); }
    finally { lock.current = false; setBusy(false); onBusy?.(false); if (input.current) input.current.value = ""; }
  }
  useFileTransfer(transferTarget || root?.closest<HTMLElement>("form") || root, {
    disabled: busy || disabled,
    label: zh ? "松开以添加研究材料" : "Drop research files to attach",
    unavailable: zh ? "请等待当前操作完成后添加文件。" : "Wait for the current operation before attaching files.",
    unreadable: zh ? "请选择文件，暂不支持文件夹。" : "Choose files; folders are not supported.",
    onFiles: upload, onError: setError,
  });
  const attachButton = <button type="button" disabled={busy || disabled} className={`research-attach-button${buttonTarget ? " file-upload-icon-btn" : ""}`} aria-label={busy ? (zh ? "上传中…" : "Uploading…") : (zh ? "添加文件" : "Attach files")} title={zh ? "添加文件" : "Attach files"} onClick={() => input.current?.click()}>
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="m21 11-8 8a6 6 0 0 1-8-8l9-9a4 4 0 0 1 6 6l-9 9a2 2 0 0 1-3-3l8-8"/></svg>
      {!buttonTarget && (busy ? (zh ? "上传中…" : "Uploading…") : (zh ? "添加文件" : "Attach files"))}
    </button>;
  return <div className="research-attachments" ref={setRoot}>
    <input ref={input} type="file" multiple hidden onChange={e => void upload(Array.from(e.target.files || []))}/>
    {buttonTarget ? createPortal(attachButton, buttonTarget) : attachButton}
    {value.map(file => <span className="research-attachment" key={file.id}>{file.name}<button type="button" disabled={busy} aria-label={(zh ? "移除 " : "Remove ") + file.name} onClick={() => onChange(value.filter(f => f.id !== file.id))}>×</button></span>)}
    {error && <span role="alert" className="mp-error">{error}</span>}
  </div>;
}
