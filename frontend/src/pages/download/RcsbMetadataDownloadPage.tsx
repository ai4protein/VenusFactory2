import { DownloadTaskPage } from "./DownloadTaskPage";
import { useLang } from "../../lib/i18n";

const STRINGS = {
  en: {
    title: "RCSB Metadata",
    subtitle: "Download annotation metadata for PDB entries from RCSB.",
    idLabel: "PDB ID",
    fileHint: "Upload a .txt list of PDB IDs (one per line)."
  },
  zh: {
    title: "RCSB 元数据",
    subtitle: "从 RCSB 下载 PDB 条目的注释元数据。",
    idLabel: "PDB ID",
    fileHint: "上传一个 .txt 文件，每行一个 PDB ID。"
  }
};

export function RcsbMetadataDownloadPage() {
  const t = useLang().t(STRINGS);
  return (
    <DownloadTaskPage
      config={{
        title: t.title,
        subtitle: t.subtitle,
        endpoint: "rcsb-metadata",
        idLabel: t.idLabel,
        idPlaceholder: "e.g., 1a0j",
        defaultId: "1a0j",
        fileHint: t.fileHint
      }}
    />
  );
}
