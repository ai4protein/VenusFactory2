import { useEffect, useState } from "react";
import { useLang } from "../lib/i18n";

const STRINGS = {
  en: { copy: "Copy", copied: "Copied", copyTextDefault: "Copy text", copyCommand: "Copy command" },
  zh: { copy: "复制", copied: "已复制", copyTextDefault: "复制文本", copyCommand: "复制命令" }
};

type CopyableTextBlockProps = {
  text: string;
  emptyText?: string;
  wrapperClassName?: string;
  preClassName?: string;
  ariaLabel?: string;
};

export function CopyableTextBlock({
  text,
  emptyText = "",
  wrapperClassName = "",
  preClassName = "",
  ariaLabel
}: CopyableTextBlockProps) {
  const t = useLang().t(STRINGS);
  const [copied, setCopied] = useState(false);
  const content = text || emptyText;

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1400);
    return () => window.clearTimeout(timer);
  }, [copied]);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className={`copyable-pre-wrap ${wrapperClassName}`.trim()}>
      <button type="button" className="copyable-pre-btn" onClick={() => void onCopy()} aria-label={ariaLabel ?? t.copyTextDefault}>
        {copied ? t.copied : t.copy}
      </button>
      <pre className={`copyable-pre ${preClassName}`.trim()}>{content}</pre>
    </div>
  );
}

type CommandPreviewCardProps = {
  command: string;
  emptyText?: string;
};

const CMD_STRINGS = {
  en: { empty: "Click Preview Command to generate CLI." },
  zh: { empty: "点击「预览命令」生成 CLI。" }
};

export function CommandPreviewCard({
  command,
  emptyText
}: CommandPreviewCardProps) {
  const t = useLang().t(STRINGS);
  const tc = useLang().t(CMD_STRINGS);
  const [copied, setCopied] = useState(false);
  const text = command || emptyText || tc.empty;

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1400);
    return () => window.clearTimeout(timer);
  }, [copied]);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="custom-command-panel">
      <div className="custom-command-toolbar">
        <button
          type="button"
          className="custom-command-copy-btn"
          onClick={() => void onCopy()}
          aria-label={t.copyCommand}
        >
          {copied ? t.copied : t.copy}
        </button>
      </div>
      <div className="custom-command-wrap">
        <pre className="copyable-pre custom-command">{text}</pre>
      </div>
    </div>
  );
}
