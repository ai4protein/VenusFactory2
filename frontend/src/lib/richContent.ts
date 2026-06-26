/**
 * Shared "rich content" extraction. Used by both the assistant-text bubble
 * (auto-render structures/images mentioned inline) and the tool-execution
 * card (parse tool outputs for file paths / urls and render appropriately).
 */

// .pdb / .cif / .mmcif / .ent (biomol) + .mol / .sdf / .mol2 (small molecule).
// All Molstar-renderable.
export const STRUCTURE_EXT_RE =
  /(?:[\w.\/~-]+\/)*[\w.-]+\.(?:pdb|cif|mmcif|ent|mol|sdf|mol2)\b/gi;

export const IMAGE_EXT_RE = /\.(?:png|jpe?g|gif|webp|bmp|svg|tiff?)$/i;

const FASTA_EXT_RE = /\.(?:fasta|fa|faa|fna|ffn|frn)$/i;
const CSV_EXT_RE = /\.(?:csv|tsv)$/i;
const HTML_EXT_RE = /\.(?:html?|htm)$/i;

// Media file extensions we know how to render. Used by both URL and
// fs-path matchers below.
const MEDIA_EXT_GROUP =
  "(?:pdb|cif|mmcif|ent|mol|sdf|mol2|png|jpe?g|gif|svg|webp|bmp|tiff?|" +
  "fasta|fa|faa|fna|ffn|frn|csv|tsv|html?|htm)";

// Match remote URLs FIRST so http(s)://host/x.png isn't truncated to /x.png.
// Allows protocol-relative `//host/...` too (AlphaFold metadata returns those).
const URL_LIKE_RE = new RegExp(
  `(?:https?:\\/\\/|\\/\\/)[^\\s\"'\`<>)\\]]+?\\.${MEDIA_EXT_GROUP}(?:\\?[^\\s\"'\`<>)\\]]*)?`,
  "gi"
);

// Local-fs paths: must start with /, ../, or ~/. Avoids matching bare
// "thing.x" inside prose. Excludes // (handled by URL_LIKE_RE).
const PATH_LIKE_RE = new RegExp(
  `(?:^|[\\s\"'\`(\\[<])((?:\\/[^\\/\\s\"'\`<>)\\]]|\\.\\.\\/|~\\/)[\\w.\\/~-]+\\.${MEDIA_EXT_GROUP})\\b`,
  "gi"
);

/** Pull .pdb/.cif paths out of free text. Dedupe and drop ones that are a
 *  suffix of another (so a bare basename doesn't double-render). */
export function extractStructurePaths(text: string): string[] {
  if (!text) return [];
  const matches = text.match(STRUCTURE_EXT_RE) || [];
  const unique = [...new Set(matches)].filter((p) => !p.startsWith("http"));
  return unique.filter(
    (p) => !unique.some((other) => other !== p && other.endsWith("/" + p))
  );
}

export type MediaRef =
  | { kind: "structure"; path: string }
  | { kind: "image"; path: string; isUrl: boolean }
  | { kind: "fasta"; path: string }
  | { kind: "csv"; path: string }
  | { kind: "html"; path: string }
  | { kind: "download"; path: string };

/** Walk an arbitrary JS value (tool-output JSON) and pull out every string
 *  that looks like a filesystem path or URL to a media file we know how to
 *  render. Dedupes; preserves insertion order. */
export function extractMediaFromValue(v: unknown, max = 32): MediaRef[] {
  const seen = new Set<string>();
  const out: MediaRef[] = [];

  const classify = (raw: string) => {
    if (out.length >= max) return;
    if (seen.has(raw)) return;
    seen.add(raw);
    const isUrl = isRemoteUrl(raw);
    // For classification, strip query string ("…/x.png?v=2" → "…/x.png")
    const base = raw.replace(/[?#].*$/, "").toLowerCase();
    if (/\.(?:pdb|cif|mmcif|ent|mol|sdf|mol2)$/i.test(base)) {
      out.push({ kind: "structure", path: raw });
    } else if (IMAGE_EXT_RE.test(base)) {
      out.push({ kind: "image", path: raw, isUrl });
    } else if (FASTA_EXT_RE.test(base)) {
      out.push({ kind: "fasta", path: raw });
    } else if (CSV_EXT_RE.test(base)) {
      out.push({ kind: "csv", path: raw });
    } else if (HTML_EXT_RE.test(base)) {
      out.push({ kind: "html", path: raw });
    }
  };

  const visit = (x: unknown) => {
    if (out.length >= max) return;
    if (x == null) return;
    if (typeof x === "string") {
      // 1) Remote URLs FIRST so e.g. https://host/x.png isn't truncated
      // to /x.png by the local-path regex. Both http(s) and protocol-
      // relative `//host/...` are accepted.
      for (const m of x.matchAll(URL_LIKE_RE)) classify(m[0]);
      // 2) Local fs paths (require explicit /, ../, or ~/ prefix).
      for (const m of x.matchAll(PATH_LIKE_RE)) classify(m[1]);
      return;
    }
    if (Array.isArray(x)) {
      x.forEach(visit);
      return;
    }
    if (typeof x === "object") {
      for (const key of Object.keys(x as Record<string, unknown>)) {
        visit((x as Record<string, unknown>)[key]);
      }
    }
  };

  visit(v);
  return out;
}

/** If `output` is a JSON string, parse it; otherwise return as-is. */
export function parseToolOutput(output: unknown): unknown {
  if (typeof output !== "string") return output;
  const s = output.trim();
  if (!s) return s;
  if (s[0] === "{" || s[0] === "[") {
    try {
      return JSON.parse(s);
    } catch {
      return output;
    }
  }
  return output;
}

/** Strip the kimi MCP namespace prefix: `mcp__venusfactory__foo` → `foo`. */
export function shortToolName(name: string): string {
  return (name || "tool").replace(/^mcp__[^_]+__/, "");
}

/** Map a path/URL to something the browser can GET inline.
 *  - `https?://...`  → return as-is (external resource)
 *  - `//host/path`   → protocol-relative URL, return as-is (browser inherits
 *                      page protocol). Some APIs (AlphaFold) return these.
 *  - filesystem path → wrap via `/api/files/inline?path=...` which enforces
 *                      containment + extension allowlist on the backend.
 */
export function fileUrlFor(path: string): string {
  if (!path) return path;
  if (/^https?:\/\//i.test(path)) return path;
  if (/^\/\/[^\/]/.test(path)) return path;   // protocol-relative URL
  return `/api/files/inline?path=${encodeURIComponent(path)}`;
}

/** Heuristic: is this string a remote URL (not a local fs path)? */
export function isRemoteUrl(s: string): boolean {
  if (!s) return false;
  return /^https?:\/\//i.test(s) || /^\/\/[^\/]/.test(s);
}
