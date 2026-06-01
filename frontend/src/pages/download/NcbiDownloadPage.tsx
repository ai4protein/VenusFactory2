import { DownloadTaskPage } from "./DownloadTaskPage";
import { useLang } from "../../lib/i18n";

const STRINGS = {
  en: {
    title: "NCBI Sequences",
    subtitle: "Download protein sequences from NCBI in FASTA format.",
    idLabel: "NCBI ID",
    fileHint: "Upload a .txt list of NCBI IDs (one per line)."
  },
  zh: {
    title: "NCBI 序列",
    subtitle: "以 FASTA 格式从 NCBI 下载蛋白序列。",
    idLabel: "NCBI ID",
    fileHint: "上传一个 .txt 文件，每行一个 NCBI ID。"
  }
};

export function NcbiDownloadPage() {
  const t = useLang().t(STRINGS);
  return (
    <DownloadTaskPage
      config={{
        title: t.title,
        subtitle: t.subtitle,
        endpoint: "ncbi",
        idLabel: t.idLabel,
        idPlaceholder: "e.g., NP_000517.1",
        defaultId: "NP_000517.1",
        supportsMerge: true,
        fileHint: t.fileHint
      }}
    />
  );
}
