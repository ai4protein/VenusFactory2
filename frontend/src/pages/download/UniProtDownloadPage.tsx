import { DownloadTaskPage } from "./DownloadTaskPage";
import { useLang } from "../../lib/i18n";

const STRINGS = {
  en: {
    title: "UniProt Sequences",
    subtitle: "Download protein sequences from UniProt in FASTA format.",
    idLabel: "UniProt ID",
    fileHint: "Upload a .txt list of UniProt IDs (one per line)."
  },
  zh: {
    title: "UniProt 序列",
    subtitle: "以 FASTA 格式从 UniProt 下载蛋白序列。",
    idLabel: "UniProt ID",
    fileHint: "上传一个 .txt 文件，每行一个 UniProt ID。"
  }
};

export function UniProtDownloadPage() {
  const t = useLang().t(STRINGS);
  return (
    <DownloadTaskPage
      config={{
        title: t.title,
        subtitle: t.subtitle,
        endpoint: "uniprot",
        idLabel: t.idLabel,
        idPlaceholder: "e.g., P00734",
        defaultId: "P00734",
        supportsMerge: true,
        fileHint: t.fileHint
      }}
    />
  );
}
