import DOMPurify from "dompurify";
import { marked, type Tokens } from "marked";
import { getManualSection, type ManualLanguage, type ManualSectionKey } from "./manualCatalog";
export { getManualSection, MANUAL_SECTIONS, langToManual } from "./manualCatalog";
export type { ManualLanguage, ManualSectionKey } from "./manualCatalog";

export type TocItem = { id: string; title: string; level: 1 | 2 | 3 };
marked.setOptions({gfm: true, breaks: true});

export async function loadManualMarkdown(sectionKey: ManualSectionKey, language: ManualLanguage): Promise<string> {
  const section = getManualSection(sectionKey);
  if (!section) return language === "Chinese" ? "# 未找到教程" : "# Guide not found";
  try {
    const response = await fetch(`/manual-docs/${section.files[language]}`, {cache: "no-store"});
    if (!response.ok || response.headers.get('content-type')?.includes('text/html')) throw new Error('unavailable');
    return await response.text();
  } catch {
    return language === "Chinese" ? "# 教程暂不可用\n\n请刷新页面重试。" : "# Guide temporarily unavailable\n\nRefresh the page to try again.";
  }
}

function resolveImagePath(src: string): string {
  if (!src) return src;
  if (src.startsWith("http://") || src.startsWith("https://") || src.startsWith("data:")) {
    return src;
  }
  if (src.startsWith("/")) {
    return src;
  }
  const cleaned = src.replace(/^\.?\//, "");
  return `/manual-docs/${cleaned}`;
}

export function renderManualHtml(markdownText: string, language: ManualLanguage = "English"): { toc: TocItem[]; html: string } {
  const tokens = marked.lexer(markdownText || "");
  const toc: TocItem[] = [];
  let headingIndex = 0;

  for (const token of tokens) {
    if (token.type === "heading" && token.depth >= 1 && token.depth <= 3) {
      toc.push({
        id: `header-${headingIndex}`,
        title: token.text,
        level: token.depth as 1 | 2 | 3
      });
      headingIndex += 1;
    }
  }

  let renderHeadingIndex = 0;
  const renderer = new marked.Renderer();

  renderer.heading = (heading: Tokens.Heading) => {
    const depth = heading.depth;
    const titleHtml = marked.Parser.parseInline(heading.tokens);
    if (depth >= 1 && depth <= 3) {
      const id = `header-${renderHeadingIndex}`;
      renderHeadingIndex += 1;
      return `<h${depth}><span id="${id}"></span>${titleHtml}</h${depth}>`;
    }
    return `<h${depth}>${titleHtml}</h${depth}>`;
  };

  renderer.link = (link: Tokens.Link) => {
    const lang = language === "Chinese" ? "zh" : "en";
    const section = link.href?.startsWith("manual:") ? getManualSection(link.href.slice(7)) : null;
    const href = section ? `/${lang}/manual/${section.key}` : (link.href || "");
    const escaped = href.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
    return `<a href="${escaped}">${marked.Parser.parseInline(link.tokens)}</a>`;
  };

  renderer.image = (image: Tokens.Image) => {
    const src = resolveImagePath(image.href || "");
    const titleAttr = image.title ? ` title="${image.title}"` : "";
    const alt = image.text || "";
    return `<img src="${src}" alt="${alt}"${titleAttr} />`;
  };

  const unsafeHtml = marked.parser(tokens, { renderer });
  const safeHtml = DOMPurify.sanitize(unsafeHtml);
  return { toc, html: safeHtml };
}
