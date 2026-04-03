import DOMPurify from "dompurify";
import { marked, type Tokens } from "marked";

export type ManualLanguage = "English" | "Chinese";

export type ManualSectionKey =
  | "report"
  | "agent"
  | "training"
  | "prediction"
  | "evaluation"
  | "quick-tools"
  | "advanced-tools"
  | "download"
  | "faq";

export type TocItem = {
  id: string;
  title: string;
  level: 1 | 2 | 3;
};

type ManualSectionDef = {
  key: ManualSectionKey;
  label: string;
  files: Record<ManualLanguage, string>;
};

const ERROR_FALLBACK_TITLE: Record<ManualSectionKey, string> = {
  report: "# Error loading Report manual",
  agent: "# Error loading Agent manual",
  training: "# Error loading manual",
  prediction: "# Error loading manual",
  evaluation: "# Error loading manual",
  "quick-tools": "# Error loading Quick Tools manual",
  "advanced-tools": "# Error loading Advanced Tools manual",
  download: "# Error loading manual",
  faq: "# FAQ"
};

export const MANUAL_SECTIONS: ManualSectionDef[] = [
  {
    key: "report",
    label: "Report",
    files: { English: "ReportManual_EN.md", Chinese: "ReportManual_CN.md" }
  },
  {
    key: "agent",
    label: "Agent",
    files: { English: "AgentManual_EN.md", Chinese: "AgentManual_CN.md" }
  },
  {
    key: "training",
    label: "Training",
    files: { English: "TrainingManual_EN.md", Chinese: "TrainingManual_ZH.md" }
  },
  {
    key: "prediction",
    label: "Prediction",
    files: { English: "PredictionManual_EN.md", Chinese: "PredictionManual_ZH.md" }
  },
  {
    key: "evaluation",
    label: "Evaluation",
    files: { English: "EvaluationManual_EN.md", Chinese: "EvaluationManual_ZH.md" }
  },
  {
    key: "quick-tools",
    label: "Quick Tools",
    files: { English: "QuickTools_EN.md", Chinese: "QuickTools_CN.md" }
  },
  {
    key: "advanced-tools",
    label: "Advanced Tools",
    files: { English: "AdvancedToolsManual_EN.md", Chinese: "AdvancedToolsManual_CN.md" }
  },
  {
    key: "download",
    label: "Download",
    files: { English: "DownloadManual_EN.md", Chinese: "DownloadManual_ZH.md" }
  },
  {
    key: "faq",
    label: "FAQ",
    files: { English: "QAManual_EN.md", Chinese: "QAManual_ZH.md" }
  }
];

const MANUAL_SECTION_MAP = new Map(MANUAL_SECTIONS.map((item) => [item.key, item]));

marked.setOptions({
  gfm: true,
  breaks: true
});

export function getManualSection(key: string | undefined): ManualSectionDef | null {
  if (!key) return null;
  return MANUAL_SECTION_MAP.get(key as ManualSectionKey) ?? null;
}

export async function loadManualMarkdown(
  sectionKey: ManualSectionKey,
  language: ManualLanguage
): Promise<string> {
  const section = MANUAL_SECTION_MAP.get(sectionKey);
  if (!section) {
    return "# Error loading manual\n\nUnknown manual section.";
  }
  const fileName = section.files[language];
  try {
    const response = await fetch(`/manual-docs/${fileName}`, {
      cache: "no-store"
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.text();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return `${ERROR_FALLBACK_TITLE[sectionKey]}\n\n${message}`;
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

export function renderManualHtml(markdownText: string): { toc: TocItem[]; html: string } {
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
