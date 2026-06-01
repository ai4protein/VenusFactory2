import { useState } from "react";
import {
  loadQuickToolDefaultExample,
  runProteinDiscoveryToolStream,
  uploadQuickToolFile
} from "../../lib/quickToolsApi";
import { QuickToolsLayout } from "./QuickToolsLayout";
import { QuickToolResultPanel } from "./QuickToolResultPanel";
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";
import { COMMON_STRINGS } from "../../lib/commonStrings";

const STRINGS = {
  en: {
    ...COMMON_STRINGS.en,
    title: "Protein Discovery (VenusMine)",
    subtitle: "Quick mode keeps only PDB input. Advanced discovery parameters use backend defaults.",
    readonlyBanner: "Online mode: protein discovery controls are view-only in this deployment.",
    pdbInput: "PDB Input",
    useExamplePdb: "Use Example PDB",
    selected: "Selected:",
    startDiscovery: "Start VenusMine Discovery",
    pdbOnlyError: "Protein Discovery only supports PDB structure input.",
    pleaseUploadPdb: "Please upload or pick a PDB file first.",
    resultTitle: "Protein Discovery Result"
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "蛋白挖掘（VenusMine）",
    subtitle: "快速模式仅保留 PDB 输入，挖掘的高级参数使用后端默认值。",
    readonlyBanner: "在线模式：当前部署下蛋白挖掘控件仅可查看。",
    pdbInput: "PDB 输入",
    useExamplePdb: "使用示例 PDB",
    selected: "已选择：",
    startDiscovery: "开始 VenusMine 挖掘",
    pdbOnlyError: "蛋白挖掘仅支持 PDB 结构输入。",
    pleaseUploadPdb: "请先上传或选择一个 PDB 文件。",
    resultTitle: "蛋白挖掘结果"
  }
};

type ProteinDiscoveryPageProps = {
  readonly?: boolean;
  workspaceEnabled?: boolean;
};

export function ProteinDiscoveryPage({ readonly = false, workspaceEnabled = false }: ProteinDiscoveryPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
  const [uploadedPath, setUploadedPath] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const data = await uploadQuickToolFile(file);
      if (data.suffix !== ".pdb") {
        throw new Error(t.pdbOnlyError);
      }
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.uploadFailed);
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadQuickToolDefaultExample("pdb");
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.loadExampleFailed);
    }
  }

  async function onRun() {
    setError("");
    setProgress(0);
    setProgressMessage(t.preparingTask);
    setRunning(true);
    try {
      if (!uploadedPath) {
        throw new Error(t.pleaseUploadPdb);
      }
      const payload = await runProteinDiscoveryToolStream(
        { pdbFile: uploadedPath },
        (evt) => {
          setProgress(evt.progress);
          setProgressMessage(evt.message);
        }
      );
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage(t.predictionDone);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.runFailed);
    } finally {
      setRunning(false);
    }
  }

  return (
    <QuickToolsLayout
      title={t.title}
      subtitle={t.subtitle}
      running={running}
      progress={progress}
      progressMessage={progressMessage || t.idle}
      left={
        <div className={`advanced-discovery-form ${readonly ? "readonly-mode" : ""}`}>
          {readonly && (
            <div className="readonly-banner" role="status" aria-live="polite">
              {t.readonlyBanner}
            </div>
          )}
          <fieldset className="readonly-fieldset advanced-discovery-fieldset" disabled={readonly}>
            <section className="custom-section-card">
              <h3>{t.pdbInput}</h3>
              <div className="custom-file-example-row upload-source-stack">
                <div className="file-source-inline">
                  <label className="left-controls custom-file-picker-field">
                    {t.selectFile}
                    <input type="file" accept=".pdb" onChange={(e) => void onUpload(e.target.files?.[0] || null)} />
                  </label>
                  <WorkspaceFilePicker
                    workspaceEnabled={workspaceEnabled}
                    disabled={running || readonly}
                    acceptedCategories={["structure"]}
                    buttonLabel={t.fromWorkspace}
                    onPick={(picked) => {
                      const selected = picked[0];
                      if (!selected || selected.suffix !== ".pdb") return;
                      setUploadedPath(selected.storage_path);
                    }}
                  />
                </div>
                <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                  {t.useExamplePdb}
                </button>
              </div>
              {uploadedPath && <div className="report-preview">{t.selected} {uploadedPath}</div>}
            </section>

            <button
              type="button"
              className="custom-btn-primary advanced-discovery-submit"
              onClick={() => void onRun()}
              disabled={running || !uploadedPath}
            >
              {running ? t.runningBtn : t.startDiscovery}
            </button>
          </fieldset>
        </div>
      }
      right={
        <QuickToolResultPanel
          title={t.resultTitle}
          resultPayload={resultPayload}
          aiSummary=""
          error={error}
        />
      }
    />
  );
}
