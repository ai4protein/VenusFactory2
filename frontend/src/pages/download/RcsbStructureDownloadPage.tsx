import { DownloadTaskPage } from "./DownloadTaskPage";
import { useLang } from "../../lib/i18n";

const STRINGS = {
  en: {
    title: "RCSB Structure",
    subtitle: "Download protein structure files from RCSB (pdb/cif).",
    idLabel: "PDB ID",
    fileHint: "Upload a .txt list of PDB IDs (one per line)."
  },
  zh: {
    title: "RCSB 结构",
    subtitle: "从 RCSB 下载蛋白结构文件（pdb / cif）。",
    idLabel: "PDB ID",
    fileHint: "上传一个 .txt 文件，每行一个 PDB ID。"
  }
};

export function RcsbStructureDownloadPage() {
  const t = useLang().t(STRINGS);
  return (
    <DownloadTaskPage
      config={{
        title: t.title,
        subtitle: t.subtitle,
        endpoint: "rcsb-structure",
        idLabel: t.idLabel,
        idPlaceholder: "e.g., 1a0j",
        defaultId: "1a0j",
        supportsFileType: true,
        showVisualization: true,
        fileHint: t.fileHint
      }}
    />
  );
}
