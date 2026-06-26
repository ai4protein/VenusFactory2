/**
 * Centralized markdown rendering.
 *
 * Uses `marked` for parsing, `highlight.js/core` (with a handpicked language
 * set to keep bundle size down) for syntax coloring inside code blocks, and
 * `DOMPurify` to sanitize the resulting HTML before injection.
 *
 * Markdown tables are wrapped in a horizontal-scroll div via a custom marked
 * extension so wide BLAST / FASTA / SARS-CoV-2 spike tables don't blow out
 * the chat column on narrow viewports.
 */
import DOMPurify from "dompurify";
import { marked } from "marked";

import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import r from "highlight.js/lib/languages/r";
import shell from "highlight.js/lib/languages/shell";
import sql from "highlight.js/lib/languages/sql";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";
import plaintext from "highlight.js/lib/languages/plaintext";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("sh", bash);
hljs.registerLanguage("shell", shell);
hljs.registerLanguage("json", json);
hljs.registerLanguage("yaml", yaml);
hljs.registerLanguage("yml", yaml);
hljs.registerLanguage("python", python);
hljs.registerLanguage("py", python);
hljs.registerLanguage("r", r);
hljs.registerLanguage("sql", sql);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("html", xml);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("md", markdown);
hljs.registerLanguage("plaintext", plaintext);
hljs.registerLanguage("text", plaintext);

let _initialized = false;
function ensureInitialized() {
  if (_initialized) return;
  _initialized = true;

  marked.setOptions({
    gfm: true,
    breaks: true,
  });

  // Override the default code renderer to (a) syntax-highlight via hljs and
  // (b) emit a wrapper carrying the language label, so a copy button can be
  // injected via the existing post-render DOM pass in ChatMessageBody.
  marked.use({
    renderer: {
      code(this: unknown, tokenOrCode: unknown, infostring?: string): string {
        // marked v15 passes a token object: { text, lang, escaped }.
        // Older code paths may pass (code, infostring) — handle both.
        let raw: string;
        let lang: string;
        if (typeof tokenOrCode === "string") {
          raw = tokenOrCode;
          lang = (infostring || "").trim().split(/\s+/)[0] || "";
        } else {
          const tok = tokenOrCode as { text?: string; lang?: string };
          raw = tok.text ?? "";
          lang = (tok.lang || "").trim().split(/\s+/)[0] || "";
        }
        const aliasMap: Record<string, string> = {
          js: "plaintext", ts: "plaintext", typescript: "plaintext",
          javascript: "plaintext", fasta: "plaintext", pdb: "plaintext",
          csv: "plaintext", tsv: "plaintext", smiles: "plaintext",
        };
        const resolved = aliasMap[lang.toLowerCase()] || lang.toLowerCase();
        let body: string;
        let langClass: string;
        if (resolved && hljs.getLanguage(resolved)) {
          try {
            body = hljs.highlight(raw, { language: resolved, ignoreIllegals: true }).value;
            langClass = `language-${resolved} hljs`;
          } catch {
            body = escapeHtml(raw);
            langClass = `language-${resolved}`;
          }
        } else {
          body = escapeHtml(raw);
          langClass = lang ? `language-${lang}` : "language-plaintext";
        }
        const langLabel = lang ? `<span class="code-lang-label">${escapeHtml(lang)}</span>` : "";
        return `<pre data-lang="${escapeHtml(lang)}">${langLabel}<code class="${langClass}">${body}</code></pre>`;
      },

      // Wrap tables in a horizontal-scroll div so wide tables don't blow out
      // the chat column. marked v15: same token-object pattern as code.
      table(this: unknown, tokenOrHeader: unknown, body?: unknown): string {
        let html: string;
        if (typeof tokenOrHeader === "string") {
          // Legacy (header, body) signature
          html = `<table><thead>${tokenOrHeader}</thead><tbody>${body ?? ""}</tbody></table>`;
        } else {
          // marked v15 token: { header, rows, align }
          const tok = tokenOrHeader as {
            header: Array<{ text: string; tokens: unknown[]; align?: string | null }>;
            rows: Array<Array<{ text: string; tokens: unknown[]; align?: string | null }>>;
          };
          const headerHtml = tok.header
            .map((cell) => `<th>${marked.parseInline(cell.text)}</th>`)
            .join("");
          const bodyHtml = tok.rows
            .map(
              (row) =>
                `<tr>${row
                  .map((cell) => `<td>${marked.parseInline(cell.text)}</td>`)
                  .join("")}</tr>`
            )
            .join("");
          html = `<table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
        }
        return `<div class="md-table-scroll">${html}</div>`;
      },
    },
  });
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function renderMarkdown(text: string): string {
  ensureInitialized();
  const html = marked.parse(text || "", { async: false }) as string;
  return DOMPurify.sanitize(html, {
    ADD_TAGS: ["img"],
    ADD_ATTR: ["src", "alt", "style", "width", "height", "loading", "data-lang"],
  });
}
