import { DownloadTaskPage } from "./DownloadTaskPage";
import { useLang } from "../../lib/i18n";

const STRINGS = {
  en: {
    title: "AlphaFold Structure",
    subtitle: "Download AlphaFold DB structure files by UniProt ID.",
    idLabel: "UniProt ID",
    fileHint: "Upload a .txt list of UniProt IDs (one per line)."
  },
  zh: {
    title: "AlphaFold 结构",
    subtitle: "按 UniProt ID 从 AlphaFold DB 下载结构文件。",
    idLabel: "UniProt ID",
    fileHint: "上传一个 .txt 文件，每行一个 UniProt ID。"
  }
};

export function AlphaFoldDownloadPage() {
  const t = useLang().t(STRINGS);
  return (
    <DownloadTaskPage
      config={{
        title: t.title,
        subtitle: t.subtitle,
        endpoint: "alphafold-structure",
        idLabel: t.idLabel,
        idPlaceholder: "e.g., P00734",
        defaultId: "P00734",
        showVisualization: true,
        fileHint: t.fileHint
      }}
    />
  );
}
