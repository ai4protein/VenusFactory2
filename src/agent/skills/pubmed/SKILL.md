---
name: pubmed
description: PubMed — NCBI's biomedical literature database (>35M citations). Keyword search inline, or batch-fetch full title + structured abstract + authors + DOI for a known list of PMIDs. Use for medical / biological literature lookup, citation resolution, or building an abstract corpus for downstream NLP. Honors NCBI_API_KEY env var.
license: Unknown
metadata:
  version: "1.0"
  skill-author: VenusFactory2 (download_pubmed_abstracts_by_pmids adapted from Google DeepMind).
---

# PubMed

## Overview

Two complementary tools: existing `query_pubmed_tool` (keyword search, returns small JSON inline) + new `download_pubmed_abstracts_by_pmids` (batch-fetch full abstracts by PMID list, one efetch XML call, saves normalized JSON to disk).

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **query_pubmed** | `query` (text), `max_results` (default 5, max 50), `max_content_length` | JSON list of paper records inline | Keyword search; small payload to agent. |
| **download_pubmed_abstracts_by_pmids** | `pmids` (list of PMID strings, max 200 per call), `out_path` (JSON file path), `timeout` (default 60s) | rich JSON envelope; saved file is `{requested_pmids, articles[]}` where each article has `pmid, title, authors, journal, year, doi, abstract` (structured abstracts join `Label: body` segments with newlines) | Batch-fetch full abstracts + metadata by PMID via efetch. |

## When to Use Each

| Goal | Tool |
|---|---|
| "Find papers about X" | `query_pubmed` |
| "Get me the full abstracts for these 50 PMIDs" | `download_pubmed_abstracts_by_pmids` |
| Build a corpus for fine-tuning a biomedical NLP model | `query_pubmed` → collect PMIDs → batch fetch via `download_pubmed_abstracts_by_pmids` |
| Resolve a citation string to a PMID | (out of scope here — use the search tool with the citation text, parse hits) |

## Output Schema for `download_pubmed_abstracts_by_pmids`

```json
{
  "requested_pmids": ["34265844", "37962427", ...],
  "articles": [
    {
      "pmid": "34265844",
      "title": "Highly accurate protein structure prediction with AlphaFold",
      "authors": ["Jumper J", "Evans R", ...],
      "journal": "Nature",
      "year": "2021",
      "doi": "10.1038/s41586-021-03819-2",
      "abstract": "Proteins are essential to life..."
    },
    ...
  ]
}
```

Structured abstracts (with `BACKGROUND:`/`METHODS:`/`RESULTS:`/`CONCLUSIONS:` sections) are joined with `\n` and each section is prefixed by its label.

## Rate Limiting

- Without `NCBI_API_KEY`: 3 req/s.
- With `NCBI_API_KEY` (free; get at https://www.ncbi.nlm.nih.gov/account/settings/): 10 req/s.
- `download_pubmed_abstracts_by_pmids` does ONE efetch for the whole batch — much friendlier than N individual fetches.
- Hard cap: 200 PMIDs per call (split into chunks if you need more).

## Common Mistakes

- **PMID as int**: the tool accepts strings; pass `"34265844"`, not `34265844`. (Int is coerced to str internally but use string defensively.)
- **>200 PMIDs in one call**: returns `ValidationError`. Split into chunks of ≤200.
- **Missing abstract**: not all PubMed records have abstracts (older / non-research / book chapters). The `abstract` field will be `null` for those.
- **Confusing PMID with PMCID**: PMID is a number (e.g. `34265844`); PMCID is `PMC<number>` (e.g. `PMC8371605`). This tool uses PMIDs.

## References

- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [Get an NCBI API key](https://www.ncbi.nlm.nih.gov/account/settings/) (free)
- [PMID vs PMCID](https://www.nlm.nih.gov/bsd/policy/dla_FAQ.html)
