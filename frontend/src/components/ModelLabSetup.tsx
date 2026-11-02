import type { ReactNode } from "react";
import { useLang } from "../lib/i18n";
import { SegmentedSwitch } from "./SegmentedSwitch";
import { WorkspaceFilePicker } from "./WorkspaceFilePicker";
import "../model-lab-setup.css";

type DataSource = "upload" | "hf_local" | "preset";

export function ModelLabDataSource({ value, onChange, action, children }: {
  value: DataSource;
  onChange: (value: DataSource) => void;
  action: ReactNode;
  children: ReactNode;
}) {
  const { lang } = useLang();
  return (
    <section className="custom-section-card lab-data-card">
      <div className="lab-card-heading">
        <h3>{lang === "zh" ? "数据集" : "Dataset"}</h3>
        {action}
      </div>
      <SegmentedSwitch
        value={value}
        onChange={onChange}
        ariaLabel={lang === "zh" ? "数据来源" : "Data source"}
        className="lab-source-switch"
        options={[
          { value: "upload", label: lang === "zh" ? "上传文件" : "Upload files" },
          { value: "hf_local", label: "Hugging Face" },
          { value: "preset", label: lang === "zh" ? "内置数据集" : "Built-in datasets" }
        ]}
      />
      <div className="lab-source-content">{children}</div>
    </section>
  );
}

export function ModelLabFileInput({ label, value, accept = ".csv,.tsv,.xlsx,.xls", disabled = false,
  workspaceEnabled, sequences = false, onUpload, onPick, onExample, onClear }: {
  label: string;
  value: string;
  accept?: string;
  disabled?: boolean;
  workspaceEnabled: boolean;
  sequences?: boolean;
  onUpload: (file: File | null) => void;
  onPick: (path: string) => void;
  onExample: () => void;
  onClear: () => void;
}) {
  const { lang } = useLang();
  const filename = value.split(/[\\/]/).pop();
  return (
    <div className="lab-file-input">
      <div className="lab-file-info">
        <span className="lab-file-label">{label}</span>
        <span className="lab-file-name" title={filename}>
          {filename || (sequences ? "CSV · Excel · FASTA" : "CSV · TSV · Excel")}
        </span>
      </div>
      <div className="lab-file-actions">
        <label className="lab-file-upload">
          <input type="file" accept={accept} aria-label={label} disabled={disabled}
            onChange={(event) => { onUpload(event.target.files?.[0] || null); event.target.value = ""; }} />
          {lang === "zh" ? "选择文件" : "Choose file"}
        </label>
        <WorkspaceFilePicker workspaceEnabled={workspaceEnabled} disabled={disabled}
          acceptedCategories={sequences ? ["table_or_text", "sequence"] : ["table_or_text"]}
          buttonLabel={lang === "zh" ? "从文件库选择" : "From Library"}
          onPick={(files) => { if (files[0]) onPick(files[0].storage_path); }} />
        <button type="button" className="custom-btn-secondary" disabled={disabled} onClick={onExample}>
          {lang === "zh" ? "使用示例" : "Use example"}
        </button>
        {value && <button type="button" className="lab-file-clear" disabled={disabled} onClick={onClear}
          aria-label={`${lang === "zh" ? "移除" : "Remove"} ${label}`}>×</button>}
      </div>
    </div>
  );
}
