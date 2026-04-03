import { DownloadTaskPage } from "./DownloadTaskPage";

export function NcbiDownloadPage() {
  return (
    <DownloadTaskPage
      config={{
        title: "NCBI Sequences",
        subtitle: "Download protein sequences from NCBI in FASTA format.",
        endpoint: "ncbi",
        idLabel: "NCBI ID",
        idPlaceholder: "e.g., NP_000517.1",
        defaultId: "NP_000517.1",
        supportsMerge: true,
        fileHint: "Upload a .txt list of NCBI IDs (one per line)."
      }}
    />
  );
}
