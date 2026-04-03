import { DownloadTaskPage } from "./DownloadTaskPage";

export function InterProDownloadPage() {
  return (
    <DownloadTaskPage
      config={{
        title: "InterPro Metadata",
        subtitle: "Download entry metadata by InterPro ID.",
        endpoint: "interpro-metadata",
        idLabel: "InterPro ID",
        idPlaceholder: "e.g., IPR000001",
        defaultId: "IPR000001",
        fileHint: "Upload a .txt list of InterPro IDs (one per line)."
      }}
    />
  );
}
