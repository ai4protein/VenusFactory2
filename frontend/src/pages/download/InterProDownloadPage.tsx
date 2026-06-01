import { DownloadTaskPage } from "./DownloadTaskPage";
import { useLang } from "../../lib/i18n";

const STRINGS = {
  en: {
    title: "InterPro Metadata",
    subtitle: "Download entry metadata by InterPro ID.",
    idLabel: "InterPro ID",
    fileHint: "Upload a .txt list of InterPro IDs (one per line)."
  },
  zh: {
    title: "InterPro 元数据",
    subtitle: "按 InterPro ID 下载条目元数据。",
    idLabel: "InterPro ID",
    fileHint: "上传一个 .txt 文件，每行一个 InterPro ID。"
  }
};

export function InterProDownloadPage() {
  const t = useLang().t(STRINGS);
  return (
    <DownloadTaskPage
      config={{
        title: t.title,
        subtitle: t.subtitle,
        endpoint: "interpro-metadata",
        idLabel: t.idLabel,
        idPlaceholder: "e.g., IPR000001",
        defaultId: "IPR000001",
        fileHint: t.fileHint
      }}
    />
  );
}
