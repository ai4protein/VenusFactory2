import { useState } from "react";
import {
  loadAdvancedDefaultExample,
  runAdvancedProteinDiscoveryStream,
  uploadAdvancedToolFile
} from "../../lib/advancedToolsApi";
import { AdvancedToolsLayout } from "./AdvancedToolsLayout";
import { AdvancedResultPanel } from "./AdvancedResultPanel";
import { WorkspaceFilePicker } from "../../components/WorkspaceFilePicker";
import { useLang } from "../../lib/i18n";
import { useDocumentMeta } from "../../lib/useDocumentMeta";
import { COMMON_STRINGS } from "../../lib/commonStrings";

const STRINGS = {
  en: {
    ...COMMON_STRINGS.en,
    title: "Protein Discovery (VenusMine)",
    subtitle: "Search and cluster structural homologs with FoldSeek and MMseqs.",
    readonlyBanner: "Online mode: protein discovery controls are view-only in this deployment.",
    pdbInputSection: "PDB Input",
    useExamplePdb: "Use Example PDB",
    advancedParamsSection: "Advanced Parameters",
    protectStart: "Protected Region Start",
    protectEnd: "Protected Region End",
    mmseqsThreads: "MMseqs Threads",
    mmseqsIterations: "MMseqs Iterations",
    mmseqsMaxSeqs: "MMseqs Max Sequences",
    clusterMinSeqId: "Cluster Min Seq Identity",
    clusterThreads: "Cluster Threads",
    topNThreshold: "Tree Top-N Threshold",
    evalueThreshold: "E-value Threshold",
    startBtn: "Start VenusMine Discovery",
    resultTitle: "Protein Discovery Result"
  },
  zh: {
    ...COMMON_STRINGS.zh,
    title: "蛋白挖掘（VenusMine）",
    subtitle: "使用 FoldSeek 和 MMseqs 搜索并聚类结构同源蛋白。",
    readonlyBanner: "在线模式：当前部署下蛋白挖掘参数仅供查看。",
    pdbInputSection: "PDB 输入",
    useExamplePdb: "使用示例 PDB",
    advancedParamsSection: "高级参数",
    protectStart: "保护区起始位点",
    protectEnd: "保护区结束位点",
    mmseqsThreads: "MMseqs 线程数",
    mmseqsIterations: "MMseqs 迭代次数",
    mmseqsMaxSeqs: "MMseqs 最大序列数",
    clusterMinSeqId: "聚类最低序列一致性",
    clusterThreads: "聚类线程数",
    topNThreshold: "进化树 Top-N 阈值",
    evalueThreshold: "E-value 阈值",
    startBtn: "开始 VenusMine 挖掘",
    resultTitle: "蛋白挖掘结果"
  }
};

type AdvancedProteinDiscoveryPageProps = {
  readonly?: boolean;
  workspaceEnabled?: boolean;
};

export function AdvancedProteinDiscoveryPage({ readonly = false, workspaceEnabled = false }: AdvancedProteinDiscoveryPageProps) {
  const t = useLang().t(STRINGS);
  useDocumentMeta({ title: `${t.title} — VenusFactory2`, description: t.subtitle });
  const [uploadedPath, setUploadedPath] = useState("");
  const [protectStart, setProtectStart] = useState(1);
  const [protectEnd, setProtectEnd] = useState(100);
  const [mmseqsThreads, setMmseqsThreads] = useState(96);
  const [mmseqsIterations, setMmseqsIterations] = useState(3);
  const [mmseqsMaxSeqs, setMmseqsMaxSeqs] = useState(100);
  const [clusterMinSeqId, setClusterMinSeqId] = useState(0.5);
  const [clusterThreads, setClusterThreads] = useState(96);
  const [topNThreshold, setTopNThreshold] = useState(10);
  const [evalueThreshold, setEvalueThreshold] = useState(1e-5);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [resultPayload, setResultPayload] = useState<Record<string, unknown> | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");

  async function onUpload(file: File | null) {
    if (!file) return;
    setError("");
    try {
      const data = await uploadAdvancedToolFile(file);
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.uploadFailed);
    }
  }

  async function onRun() {
    setError("");
    setProgress(0);
    setProgressMessage(t.preparingTask);
    setRunning(true);
    try {
      const payload = await runAdvancedProteinDiscoveryStream({
        pdb_file: uploadedPath,
        protect_start: protectStart,
        protect_end: protectEnd,
        mmseqs_threads: mmseqsThreads,
        mmseqs_iterations: mmseqsIterations,
        mmseqs_max_seqs: mmseqsMaxSeqs,
        cluster_min_seq_id: clusterMinSeqId,
        cluster_threads: clusterThreads,
        top_n_threshold: topNThreshold,
        evalue_threshold: evalueThreshold
      }, (evt) => {
        setProgress(evt.progress);
        setProgressMessage(evt.message);
      });
      setResultPayload(payload);
      setProgress(1);
      setProgressMessage(t.predictionDone);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.runFailed);
    } finally {
      setRunning(false);
    }
  }

  async function onUseExample() {
    setError("");
    try {
      const data = await loadAdvancedDefaultExample("pdb");
      setUploadedPath(data.file_path);
    } catch (err) {
      setError(err instanceof Error ? err.message : t.loadExampleFailed);
    }
  }

  return (
    <AdvancedToolsLayout
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
              <h3>{t.pdbInputSection}</h3>
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
                      if (!selected) return;
                      setUploadedPath(selected.storage_path);
                    }}
                  />
                </div>
                <button type="button" className="custom-btn-secondary" onClick={() => void onUseExample()}>
                  {t.useExamplePdb}
                </button>
              </div>
              {uploadedPath && <div className="report-preview">{t.uploaded} {uploadedPath}</div>}
            </section>

            <section className="custom-section-card">
              <h3>{t.advancedParamsSection}</h3>
              <label className="left-controls">
                {t.protectStart}
                <input type="number" value={protectStart} onChange={(e) => setProtectStart(Number(e.target.value) || 1)} />
              </label>
              <label className="left-controls">
                {t.protectEnd}
                <input type="number" value={protectEnd} onChange={(e) => setProtectEnd(Number(e.target.value) || 100)} />
              </label>
              <label className="left-controls">
                {t.mmseqsThreads}
                <input type="number" value={mmseqsThreads} onChange={(e) => setMmseqsThreads(Number(e.target.value) || 1)} />
              </label>
              <label className="left-controls">
                {t.mmseqsIterations}
                <input
                  type="number"
                  value={mmseqsIterations}
                  onChange={(e) => setMmseqsIterations(Number(e.target.value) || 1)}
                />
              </label>
              <label className="left-controls">
                {t.mmseqsMaxSeqs}
                <input type="number" value={mmseqsMaxSeqs} onChange={(e) => setMmseqsMaxSeqs(Number(e.target.value) || 1)} />
              </label>
              <label className="left-controls">
                {t.clusterMinSeqId}
                <input
                  type="number"
                  step="0.01"
                  value={clusterMinSeqId}
                  onChange={(e) => setClusterMinSeqId(Number(e.target.value) || 0.5)}
                />
              </label>
              <label className="left-controls">
                {t.clusterThreads}
                <input type="number" value={clusterThreads} onChange={(e) => setClusterThreads(Number(e.target.value) || 1)} />
              </label>
              <label className="left-controls">
                {t.topNThreshold}
                <input type="number" value={topNThreshold} onChange={(e) => setTopNThreshold(Number(e.target.value) || 10)} />
              </label>
              <label className="left-controls">
                {t.evalueThreshold}
                <input
                  type="number"
                  step="0.000001"
                  value={evalueThreshold}
                  onChange={(e) => setEvalueThreshold(Number(e.target.value) || 0.00001)}
                />
              </label>
            </section>

            <button type="button" className="custom-btn-primary advanced-discovery-submit" onClick={() => void onRun()} disabled={running}>
              {running ? t.runningBtn : t.startBtn}
            </button>
          </fieldset>
        </div>
      }
      right={
        <AdvancedResultPanel
          title={t.resultTitle}
          resultPayload={resultPayload}
          aiSummary=""
          error={error}
          showSummaryTab={false}
          enableHeatmapTab={false}
          readonly={readonly}
        />
      }
    />
  );
}
