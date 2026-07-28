---
name: openalex
description: OpenAlex — free, comprehensive scholarly graph (works, authors, sources/journals, institutions, topics, concepts, funders). Search papers by keyword/filter/sort, fetch a single entity by ID, look up author profiles, institutions, citation networks. Use whenever the user asks about papers, citation counts, author publications, institution outputs, or research topics. Lower-friction alternative to Semantic Scholar / Web of Science / Scopus.
license: CC0 (OpenAlex data) / Apache-2.0 (adapted from google-deepmind/science-skills)
metadata:
  version: "1.0"
  skill-author: VenusFactory2 (adapted from Google DeepMind)
---

# OpenAlex

## Overview

OpenAlex is a free index of >250M scholarly works built atop the Microsoft Academic Graph's successor. No API key required for the polite pool. Optional `OPENALEX_EMAIL` / `USER_EMAIL` env var → priority routing; `OPENALEX_API_KEY` env var → premium pool with higher limits.

## Project Tools (VenusFactory2)

| Tool | Args | Returns | Description |
|------|------|---------|--------------|
| **download_openalex_entries_by_query** | `entity_type` (required: `works` \| `authors` \| `sources` \| `institutions` \| `topics` \| `concepts` \| `publishers` \| `funders` \| ...), `out_dir`, `search` (free-text keyword), `filter_expr` (OpenAlex filter syntax, e.g. `authorships.author.id:A123,publication_year:>2020`), `sort` (e.g. `cited_by_count:desc`), `per_page` (default 25, max 200), `page` (default 1), `timeout` | rich JSON envelope; saved JSON page at `file_info.file_path`; `biological_metadata.total_count` + `result_count` + `next_page` hint | Search OpenAlex. |
| **download_openalex_entry_by_id** | `entity_type`, `openalex_id` (e.g. `W3177828909`, `A5089215617`, or full openalex.org URL), `out_dir`, `timeout` | rich JSON envelope; full entity JSON at `file_info.file_path`; `biological_metadata` extracts `display_name`, `cited_by_count`, `publication_year`, `doi` | Fetch a single entity by ID. |

## When to Use This Skill

- "Find papers about X" / "what's the most cited paper on Y" → search `works` with `sort=cited_by_count:desc`
- "Who is the most prolific author in topic X" → search `authors` with `filter_expr=topics.id:T...`
- "Get me the full record for [DOI/work ID]" → `download_openalex_entry_by_id` on `works`
- "What's institution X publishing about lately" → search `works` with `filter_expr=authorships.institutions.id:I...,publication_year:2024`

## Filter Syntax Cheatsheet

Filters are comma-separated for AND, `|` inside a value for OR. Examples:

| Filter | Effect |
|---|---|
| `publication_year:2023` | exact year |
| `publication_year:>2020` | year > 2020 |
| `from_publication_date:2023-01-01` | inclusive lower bound |
| `cited_by_count:>100` | well-cited papers |
| `authorships.author.id:A5089215617` | papers by a given author |
| `authorships.institutions.id:I4210090411` | papers from a given institution (`I4210090411` = DeepMind) |
| `topics.id:T11444` | papers in a specific topic |
| `is_oa:true` | open access only |
| `type:journal-article` | filter by work type |
| `doi:10.1038/s41586-021-03819-2` | exact DOI lookup |

## ID Prefixes

- `W` = Work (paper)
- `A` = Author
- `S` = Source (journal/conference/book/repo)
- `I` = Institution
- `T` = Topic
- `C` = Concept
- `P` = Publisher
- `F` = Funder

## Pagination

- Set `per_page` ≤ 200 and walk `page=1, 2, ...` until `biological_metadata.next_page` is `None`.
- For deep paging (>10k results), use OpenAlex's cursor pagination instead — out of scope for this tool; download the first page, inspect `meta.next_cursor`, then drive the cursor manually.

## Common Mistakes

- **`per_page=1000`**: max is 200. Set to 200 then paginate.
- **Wrong entity prefix**: `download_openalex_entry_by_id(entity_type='works', openalex_id='A123...')` will 404 — the prefix `A` says author, not work.
- **Filtering without an entity context**: `filter_expr=author.id:A123` makes no sense for `entity_type=authors` (use `id:A123` instead).
- **Free-text vs filter confusion**: `search="cited_by_count desc"` won't sort — use the `sort` parameter.

## References

- [OpenAlex docs](https://docs.openalex.org/)
- [Entity-type filter ref](https://docs.openalex.org/api-entities/works/filter-works) (one page per entity type)
- [Get an OpenAlex API key](https://openalex.org/) (optional)
