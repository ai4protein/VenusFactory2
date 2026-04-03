import { DownloadTaskPage } from "./DownloadTaskPage";

export function AlphaFoldDownloadPage() {
  return (
    <DownloadTaskPage
      config={{
        title: "AlphaFold Structure",
        subtitle: "Download AlphaFold DB structure files by UniProt ID.",
        endpoint: "alphafold-structure",
        idLabel: "UniProt ID",
        idPlaceholder: "e.g., P00734",
        defaultId: "P00734",
        showVisualization: true,
        fileHint: "Upload a .txt list of UniProt IDs (one per line)."
      }}
    />
  );
}
