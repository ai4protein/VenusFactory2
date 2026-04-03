import { DownloadTaskPage } from "./DownloadTaskPage";

export function RcsbMetadataDownloadPage() {
  return (
    <DownloadTaskPage
      config={{
        title: "RCSB Metadata",
        subtitle: "Download annotation metadata for PDB entries from RCSB.",
        endpoint: "rcsb-metadata",
        idLabel: "PDB ID",
        idPlaceholder: "e.g., 1a0j",
        defaultId: "1a0j",
        fileHint: "Upload a .txt list of PDB IDs (one per line)."
      }}
    />
  );
}
