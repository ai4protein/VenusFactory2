<!-- Compatibility view; edit en/researcher.md -->

# Researcher: study a protein in depth

## Provide the target and focus

Choose Researcher in Chat. Supply a protein identifier and species, sequence or structure. Use the message field for additional research instructions: therapeutic relevance, enzyme engineering, binding partners, mutation interpretation or another focus. Add papers, tables, structures and notes through the composer upload button.

> Research human carbonic anhydrase II. Cover catalytic mechanism, structures, inhibitors, selectivity and engineering opportunities. Use my assay table as supporting context.

## Choose analyses and start

Select an available free conversational model online. Choose analyses relevant to your inputs and submit. Sequence homology requires a sequence; structural similarity requires a structure. Tools and models that are unavailable in this deployment cannot be selected or invoked by Chat.

Researcher collects evidence across multiple topics and writes at least five specialist subsections. Relevant topics can include identity and function, sequences and evolution, structure and domains, interactions and pathways, properties and predictions, disease or application context, and research opportunities. Coverage depends on the input and available evidence.

## Follow the report

Watch progress and intermediate results during generation. Long remote jobs may remain pending while public services process them. You can leave and return to the same research entry. Use the section navigation to move through a long report and the file panel to open source artifacts and downloads.

## Continue or download

If the selected model service rejects requests because of quota, access or rate limits, further model requests stop. Collected evidence and existing drafts remain available for download and continuation after service recovery.

If a chapter or the final review is unfinished and the request budget remains, use **Continue research** in the same report. Completed chapters are retained; the unfinished sections are retried. Earlier drafts remain saved.

Choose **Research package (.zip)** from Download to collect the report, chapter drafts, references, evidence excerpts, analysis results and supplied sequence or structure. Uploaded papers and tables are included as extracted text; their original files remain in the conversation. The package records incomplete sections and a file checksum manifest.

For identifier-based research, the package also includes the exact reference sequence or selected PDB chain used, its source record and retrieval time. Continuing a saved research run reuses those inputs.

## Interpret the evidence

Check source links, accession IDs, analysis inputs, model names and caveats. Database annotations, literature claims, computational predictions and proposed experiments are different evidence types. Missing or failed analyses should remain identified as gaps, not converted into positive findings.

A report does not train or publish a new model by itself. Use [Scientist](./en/scientist.md) for that purpose, or [Agent](./en/agent.md) for a focused follow-up calculation.

## PDB structures and KEGG pathways

Enter a PDB ID such as `4HHB`, or `4HHB:A` for a specific chain. Choose the protein chain from the results; Researcher retrieves its coordinates automatically. Sequence calculations use observed residues of that chain, which can differ from the full UniProt reference protein.

Select **KEGG pathways & function** to retrieve mapped genes, pathways, orthology and modules. KEGG requires an explicit UniProt mapping; an unmapped sequence is not assigned pathways by name alone. Reports retain identifiers and source links. Agent and Scientist can also discover the `query_kegg` tool for search, entry lookup, UniProt conversion and linked annotations.

## Human target evidence

For an unambiguous human gene mapping, Researcher can collect Open Targets disease associations, drug candidates, expression, genetic constraint and tractability, according to the selected analyses. Each specialist receives the relevant evidence; the package retains the collected records and database version. Genetic constraint describes a gene, rather than the measured effect of a particular mutation. Agent and Scientist can access the same resources through the Open Targets tool.

## Source review

After drafting, a separate reviewer checks the claims against passages from the collected sources. Chapters with unsupported claims are revised within the remaining request budget and reviewed again. The final synthesis is checked separately. Until these checks finish, the report remains incomplete.

The research package includes the assessments, quoted evidence and original-source locations in verification/. Earlier drafts are retained in drafts/. Continuing a report reuses unchanged assessments and checks revised text or changed sources again. These checks assess support in the collected evidence; they do not replace experimental validation.

## Available prediction models

Researcher uses enabled platform tools and Factory-trained models. RSI projects are browsed and downloaded separately from [RSI Plaza](./en/rsi-plaza.md); Researcher does not execute them.
