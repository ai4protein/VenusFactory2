<!-- Compatibility view; edit en/data.md -->

# Retrieve data and evidence

## Ask in Chat

Download is a tool capability, rather than a separate sidebar page. In Agent, identify the protein, organism and requested data. Ask for source accessions and saved files, for example:

> Retrieve the reviewed UniProt sequence and available PDB structures for human lysozyme. Keep the accession IDs and save the sequence as FASTA.

Registered tools can cover UniProt, NCBI, RCSB, AlphaFold DB, InterPro and other enabled services. Web and literature search uses configured providers such as Tavily or SerpAPI; you do not need to paste a search-provider key into Chat.

## Choose a database

| What you need | Available queries |
| --- | --- |
| Target–disease links and drug candidates | **Open Targets**: search a gene, disease or drug name; inspect associations, tractability and mechanisms |
| Gene IDs and cross-species context | **Ensembl**: resolve a symbol with its species, retrieve annotations, cross-references, orthologues and sequences |
| Gene expression and enrichment | **gget**: Ensembl/UniProt information, sequences, PDB metadata, ARCHS4 expression, Bgee orthology, Open Targets and Enrichr |
| Papers and abstracts | **Europe PMC**: abstracts, DOI/PMID, publication year, open-access and retraction metadata |
| Compound information | **PubChem**: names, CID/SMILES, formula, mass and physicochemical descriptors |

For example:

> Resolve human BRCA2, summarize its Open Targets disease associations and drug candidates, and include source links.

> Find the mouse orthologue of human BRCA2 and retrieve its protein sequence.

> Find recent PET hydrolase papers in Europe PMC and compare the reported assays.

> Retrieve caffeine from PubChem, including its SMILES and molecular properties.

These database queries do not need a language-model or search-provider key of their own.
They work in local and online modes when enabled in Administration. Agent and Scientist
can discover them by task. Researcher uses the same Europe PMC tool for literature;
when UniProt gives an unambiguous Ensembl gene mapping it also collects gene annotations,
and for human proteins adds relevant Open Targets associations and drug candidates.

Open Targets covers human therapeutic targets. Resolve names before comparing records;
gene symbols can differ between species. Database associations do not establish causality,
and PubChem descriptors are not ADMET or binding predictions. Results retain source links,
retrieval time and release information where available. Use returned page/cursor information
to continue a search. gget exposes selected public-query modules; large dataset downloads
and paid AI modules are not part of this tool.

## Sequence and structure search

NCBI BLAST searches sequence similarity. Remote Foldseek searches structural similarity. gget adapters provide supported gene/sequence and annotation queries. Supply the right identifier, species, sequence or structure; a protein name alone may be ambiguous.

Remote services can return a job identifier while processing. Continue checking that identifier rather than repeatedly submitting the same search. Public server rate limits and outages are independent of platform availability.

## Platform capability boundaries

Use the current tool/model catalog to check what is enabled in this deployment. Database retrieval, remote searches, model inference and training are different operations.

- **AlphaFold DB** retrieves existing predicted structures and confidence records. VenusFactory does not expose an AlphaFold inference engine for new structure prediction.
- **ColabFold sequence search**, when enabled, queries an external public service. It is not a local MMseqs2 installation or an arbitrary MMseqs2 command interface.
- **NCBI BLAST and remote Foldseek** submit searches to their respective public servers. Availability and completion depend on those servers.
- **Structure-conditioned mutation scoring** evaluates mutations using an input structure. It does not generate that structure. pLDDT/PAE analysis interprets existing confidence values.
- A model is usable only when its adapter is configured and enabled. A mention in a paper, tutorial or previous answer does not establish availability.

## Keep provenance

Keep database accessions, source links, retrieval context and relevant versions with downloaded files. Search snippets are not full papers, and a homology hit is not experimental confirmation of function. For a comprehensive synthesis, use [Researcher](./en/researcher.md).

The analysis sandbox has no network access. Agent uses registered download/search tools first, then reads the saved files for local analysis. Local VenusMine/Foldseek discovery is distinct from the remote Foldseek service and remains unavailable online.
