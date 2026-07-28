---
name: biorxiv
description: bioRxiv & medRxiv — biology / medicine preprint servers. Search by keyword + date window, fetch a specific preprint's full metadata (with all version history + abstract + JATS XML link) by DOI. Use for cutting-edge biology/medicine work that hasn't gone through peer review yet, or to find the original preprint version of a paper you only have the DOI for.
license: Unknown (bioRxiv content licensed by individual authors)
metadata:
  version: "1.0"
  skill-author: VenusFactory2 (download_biorxiv_by_doi adapted from Google DeepMind).
---

# bioRxiv & medRxiv

## Overview

Two tools: existing `query_biorxiv_tool` (keyword/category date-window search) + new `download_biorxiv_by_doi` (fetch one preprint by DOI with all metadata, including version history).

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **query_biorxiv** | `query` (category or keyword), `max_results` (default 5, max 50), `days` (default 30; date window) | JSON list of paper records inline | Browse recent preprints in a category. |
| **download_biorxiv_by_doi** | `doi` (bare `10.1101/...` or DOI URL), `out_dir`, `server` (`biorxiv` \| `medrxiv`; default `biorxiv`), `include_abstract` (default True), `timeout` (default 30s) | rich JSON envelope; full metadata + version list at `file_info.file_path`; `biological_metadata` extracts title/authors/date/category/license/jatsxml URL | Fetch one preprint by DOI. |

## When to Use Each

| Goal | Tool |
|---|---|
| "Latest bioinformatics preprints" | `query_biorxiv(query="bioinformatics", days=7)` |
| "Get me preprint X by DOI" | `download_biorxiv_by_doi` |
| Resolve a published paper back to its preprint version | `download_biorxiv_by_doi(doi=<published-DOI>)` (if a preprint exists with that DOI) |
| Get JATS XML for ML training data | `download_biorxiv_by_doi`, then GET the `jatsxml` URL from the metadata |

## Server Choice

- **biorxiv** (default) — biology preprints (>200K papers, since 2013)
- **medrxiv** — medicine / health-sciences preprints (>40K papers, since 2019)

If a DOI isn't found on biorxiv, try medrxiv (they share `10.1101/` DOI prefix).

## Version Handling

`download_biorxiv_by_doi` returns *all* versions in `latest_versions[*]`. The top-level `latest` field is the most recent version (newest `version` number). Use `latest.jatsxml` for full-text XML, `latest.date` for publication date.

## Common Mistakes

- **Wrong server**: if you get `NotFound`, try `server="medrxiv"` (or vice versa).
- **DOI URL not normalized**: the tool strips `https://doi.org/`, `http://doi.org/`, `doi:` prefixes — fine to paste a URL.
- **Trying to download the PDF directly**: this tool returns the *metadata JSON* (which includes a PDF URL). To download the PDF, follow the URL via `WebFetch` or `requests`.
- **Old DOI on a withdrawn preprint**: the API may still return metadata with `type="withdrawn"`. Check `category` / `type` fields if integrity matters.

## References

- [bioRxiv API docs](https://api.biorxiv.org/)
- [bioRxiv home](https://www.biorxiv.org/) / [medRxiv home](https://www.medrxiv.org/)
