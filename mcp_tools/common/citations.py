"""Plain text evidence formatting for agent prompts and reports."""
def _format_literature_citations(refs: list, max_n: int = 5) -> list:
    """Format literature references as [n] [title](url) for PI citations (Markdown link format).
    Only shows available fields, no NA or empty values.
    """
    out = []
    for i, r in enumerate((refs or [])[:max_n], 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or r.get("citation") or "No title").strip()

        # Build metadata parts, only include non-empty values
        authors = r.get("authors") or r.get("author") or ""
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors[:5] if a)
        year = r.get("year") or r.get("published") or ""

        # Build metadata string with only available fields
        meta_parts = []
        if authors:
            meta_parts.append(authors)
        if year:
            meta_parts.append(str(year))
        meta_str = f" — {', '.join(meta_parts)}" if meta_parts else ""

        url = r.get("url") or r.get("link") or ""
        if url:
            out.append(f"[{i}] [{title}]({url}){meta_str}")
        else:
            out.append(f"[{i}] {title}{meta_str}")
    return out

def _format_literature_for_reading(refs: list, max_n: int = 5, abstract_max: int = 400) -> list:
    """Format literature with abstract so PI can read and cite. Each item: [title](url) — authors, year.
    Only shows available fields, no NA or empty values.
    """
    out = []
    for i, r in enumerate((refs or [])[:max_n], 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or r.get("citation") or "No title").strip()

        # Build metadata parts, only include non-empty values
        authors = r.get("authors") or r.get("author") or ""
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors[:5] if a)
        year = r.get("year") or r.get("published") or ""

        meta_parts = []
        if authors:
            meta_parts.append(authors)
        if year:
            meta_parts.append(str(year))
        meta_str = f" — {', '.join(meta_parts)}" if meta_parts else ""

        url = r.get("url") or r.get("link") or ""
        if url:
            line = f"[{title}]({url}){meta_str}"
        else:
            line = f"{title}{meta_str}"

        abstract = (r.get("abstract") or "").strip()
        if abstract:
            ab = abstract[:abstract_max] + ("…" if len(abstract) > abstract_max else "")
            line += f"\n  **Abstract:** {ab}"
        out.append(line)
    return out

def _format_web_citations(results: list, max_n: int = 5) -> list:
    """Format web search results as [n] [title](url) for PI citations (Markdown link format)."""
    out = []
    for i, r in enumerate((results or [])[:max_n], 1):
        if isinstance(r, dict):
            title = (r.get("title") or r.get("snippet") or str(r)).strip()
            url = r.get("url") or ""
            if url:
                out.append(f"[{i}] [{title}]({url})")
            else:
                out.append(f"[{i}] {title}")
        else:
            out.append(f"[{i}] {str(r)}")
    return out

def _format_web_for_reading(results: list, max_n: int = 5, snippet_max: int = 300) -> list:
    """Format web results with snippet so PI can read and cite. Each item: [n] [title](url)."""
    out = []
    for i, r in enumerate((results or [])[:max_n], 1):
        if isinstance(r, dict):
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            url = r.get("url") or ""
            if url:
                line = f"[{title or 'Link'}]({url})"
            else:
                line = f"{title or str(r)}"
            if snippet:
                sn = snippet[:snippet_max] + ("…" if len(snippet) > snippet_max else "")
                line += f"\n  **Snippet:** {sn}"
            out.append(line)
        else:
            out.append(f"[{i}] {str(r)}")
    return out

def _format_dataset_citations(datasets: list, max_n: int = 5) -> list:
    """Format dataset search results as [n] [title](url) for PI citations (Markdown link format)."""
    out = []
    for i, d in enumerate((datasets or [])[:max_n], 1):
        if not isinstance(d, dict):
            continue
        title = (d.get("title") or "Untitled").strip()
        url = d.get("url") or ""
        src = d.get("source") or ""
        if url:
            out.append(f"[{i}] [{title}]({url})" + (f" — Source: {src}" if src else ""))
        else:
            out.append(f"[{i}] {title}" + (f" — Source: {src}" if src else ""))
    return out
