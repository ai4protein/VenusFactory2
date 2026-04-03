import { useEffect, useState } from "react";

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
  ariaLabel = "Copy text"
}: CopyableTextBlockProps) {
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
      <button type="button" className="copyable-pre-btn" onClick={() => void onCopy()} aria-label={ariaLabel}>
        {copied ? "Copied" : "Copy"}
      </button>
      <pre className={`copyable-pre ${preClassName}`.trim()}>{content}</pre>
    </div>
  );
}

type CommandPreviewCardProps = {
  command: string;
  emptyText?: string;
};

export function CommandPreviewCard({
  command,
  emptyText = "Click Preview Command to generate CLI."
}: CommandPreviewCardProps) {
  const [copied, setCopied] = useState(false);
  const text = command || emptyText;

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
          aria-label="Copy command"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="custom-command-wrap">
        <pre className="copyable-pre custom-command">{text}</pre>
      </div>
    </div>
  );
}
