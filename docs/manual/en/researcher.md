# Researcher: study a protein in depth

## Provide the target and focus

Choose Researcher in Chat. Supply a protein identifier and species, sequence or structure. Use the message field for additional research instructions: therapeutic relevance, enzyme engineering, binding partners, mutation interpretation or another focus. Add papers, tables, structures and notes through the composer upload button.

> Research human carbonic anhydrase II. Cover catalytic mechanism, structures, inhibitors, selectivity and engineering opportunities. Use my assay table as supporting context.

## Choose analyses and start

Select an available free conversational model online. Choose analyses relevant to your inputs and submit. Sequence homology requires a sequence; structural similarity requires a structure. Tools and models that are unavailable in this deployment cannot be selected or invoked by Chat.

Researcher collects evidence across multiple topics and writes at least five specialist subsections. Relevant topics can include identity and function, sequences and evolution, structure and domains, interactions and pathways, properties and predictions, disease or application context, and research opportunities. Coverage depends on the input and available evidence.

## Follow the report

Watch progress and intermediate results during generation. Long remote jobs may remain pending while public services process them. You can leave and return to the same research entry. Use the section navigation to move through a long report and the file panel to open source artifacts and downloads.

## Interpret the evidence

Check source links, accession IDs, analysis inputs, model names and caveats. Database annotations, literature claims, computational predictions and proposed experiments are different evidence types. Missing or failed analyses should remain identified as gaps, not converted into positive findings.

A report does not train or publish a new model by itself. Use [Scientist](manual:scientist) for that purpose, or [Agent](manual:agent) for a focused follow-up calculation.

## PDB structures and KEGG pathways

Enter a PDB ID such as `4HHB`, or `4HHB:A` for a specific chain. Choose the protein chain from the results; Researcher retrieves its coordinates automatically. Sequence calculations use observed residues of that chain, which can differ from the full UniProt reference protein.

Select **KEGG pathways & function** to retrieve mapped genes, pathways, orthology and modules. KEGG requires an explicit UniProt mapping; an unmapped sequence is not assigned pathways by name alone. Reports retain identifiers and source links. Agent and Scientist can also discover the `query_kegg` tool for search, entry lookup, UniProt conversion and linked annotations.
