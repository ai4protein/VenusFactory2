import { DownloadTaskPage } from "./DownloadTaskPage";

export function UniProtDownloadPage() {
  return (
    <DownloadTaskPage
      config={{
        title: "UniProt Sequences",
        subtitle: "Download protein sequences from UniProt in FASTA format.",
        endpoint: "uniprot",
        idLabel: "UniProt ID",
        idPlaceholder: "e.g., P00734",
        defaultId: "P00734",
        supportsMerge: true,
        fileHint: "Upload a .txt list of UniProt IDs (one per line)."
      }}
    />
  );
}
