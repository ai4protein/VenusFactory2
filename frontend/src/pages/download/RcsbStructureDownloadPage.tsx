import { DownloadTaskPage } from "./DownloadTaskPage";

export function RcsbStructureDownloadPage() {
  return (
    <DownloadTaskPage
      config={{
        title: "RCSB Structure",
        subtitle: "Download protein structure files from RCSB (pdb/cif).",
        endpoint: "rcsb-structure",
        idLabel: "PDB ID",
        idPlaceholder: "e.g., 1a0j",
        defaultId: "1a0j",
        supportsFileType: true,
        showVisualization: true,
        fileHint: "Upload a .txt list of PDB IDs (one per line)."
      }}
    />
  );
}
