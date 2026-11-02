import { useLang } from "../lib/i18n";
import { openStructure } from "../lib/structurePreview";
export function StructureFileButton({ filePath, label }: { filePath: string; label?: string }) {
  const { lang } = useLang();
  const name = label || filePath.split(/[?#]/)[0].split("/").pop() || filePath;
  return <button type="button" className="structure-file-button" onClick={() => openStructure({ path: filePath, name })} title={name}>
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m8 7 8 3-4 9M8 7l4 12"/><circle cx="8" cy="7" r="3"/><circle cx="17" cy="10" r="3"/><circle cx="12" cy="19" r="2"/></svg>
    <span>{name}</span><small>{lang === "zh" ? "查看结构" : "View structure"}</small>
  </button>;
}
