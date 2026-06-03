---
name: arxiv
description: arXiv preprint server — keyword search the official API and download papers as PDF / HTML / source tarball. Use whenever the user mentions an arXiv ID (e.g. 2106.04559) or wants preprints on a topic in CS / physics / math / quantitative biology. For biomedical preprints specifically, prefer biorxiv skill.
license: Unknown (arXiv content under various licenses, see paper metadata)
metadata:
    skill-author: VenusFactory2.
---

# arXiv

## Overview

Two complementary tools: existing `query_arxiv_tool` (text search, returns JSON list inline) + new `download_arxiv_paper_by_id` (fetches the actual paper as PDF / HTML / source tarball to disk).

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **query_arxiv** | `query` (text), `max_results` (default 5, max 50), `max_content_length` (default 10000) | JSON list of paper records (title, abstract, arxiv_id, authors, etc.) inline | Search-only; small JSON returned to agent context. |
| **download_arxiv_paper_by_id** | `arxiv_id` (e.g. `2106.04559`, `2106.04559v2`, `hep-th/9510017`), `out_dir`, `format` (`pdf` \| `html` \| `source`; default `pdb`), `timeout` (default 60s) | rich JSON envelope; file at `file_info.file_path` (`arxiv_<id>.pdf` etc.) | Download the actual paper. |

## When to Use Each

| Goal | Tool |
|---|---|
| "Find papers about X" | `query_arxiv` |
| User provided an arxiv id and wants the PDF | `download_arxiv_paper_by_id` |
| Build a literature review pipeline (search → read PDFs) | `query_arxiv` → `download_arxiv_paper_by_id` for each result |
| Get the LaTeX source / figures for a paper | `download_arxiv_paper_by_id` with `format=source` (tar.gz) |

## ID Formats Accepted

- Modern: `2106.04559`, `2106.04559v2` (with version)
- Old-style: `hep-th/9510017`, `cond-mat/0411174`
- The tool strips the `arxiv:` / `arXiv:` prefix if present.

## Format Details

| `format` | URL pattern | When to use |
|---|---|---|
| `pdf` (default) | `https://arxiv.org/pdf/<id>.pdf` | Read the paper |
| `html` | `https://arxiv.org/html/<id>` | Modern papers only (post-2023); fails 404 on older papers |
| `source` | `https://export.arxiv.org/e-print/<id>` | LaTeX source as `.tar.gz` (for reproducing figures, extracting data, etc.) |

## Rate Limiting

- arXiv asks for ~3-second spacing between requests. The tool doesn't enforce this internally; if you fire many in a loop, sleep yourself.
- Bulk downloads should use [the arXiv full-text bulk dataset on AWS S3](https://info.arxiv.org/help/bulk_data_s3.html), not this tool.

## Common Mistakes

- **HTML format on an old paper**: 404. Fall back to `pdf`.
- **PDF response is HTML**: arXiv occasionally serves an error page with 200 OK; the tool checks `%PDF` magic bytes and returns `DownloadError` if the bytes aren't a real PDF.
- **No abstract from `download_arxiv_paper_by_id`**: this tool only downloads; for abstract / metadata, use `query_arxiv` (or fetch the abstract page via `WebFetch`).

## References

- [arXiv API user manual](https://info.arxiv.org/help/api/user-manual.html)
- [arXiv Terms of Use](https://info.arxiv.org/help/api/tou.html)
