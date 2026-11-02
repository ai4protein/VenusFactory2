import { useEffect, useId, useRef, useState, type ButtonHTMLAttributes, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { historyTime } from "../lib/historyPreferences";
import "./HistorySessionButton.css";
import "./history-running.css";

export function HistorySessionButton({ date, lang, label, children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & {
  date: string | number; lang: string; label: string;
}) {
  const id = useId(); const button = useRef<HTMLButtonElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const [placement, setPlacement] = useState<CSSProperties | null>(null);
  const time = historyTime(date, lang);
  function hide() { clearTimeout(timer.current); setPlacement(null); }
  function show() {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const rect = button.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(280, window.innerWidth - 24);
      const fitsRight = rect.right + width + 20 < window.innerWidth;
      setPlacement({ width, left: fitsRight ? rect.right + 10 : Math.max(12, Math.min(rect.left, window.innerWidth - width - 12)),
        top: Math.max(12, Math.min(fitsRight ? rect.top : rect.bottom + 6, window.innerHeight - 116)) });
    }, 350);
  }
  useEffect(() => {
    window.addEventListener("scroll", hide, true); window.addEventListener("resize", hide);
    return () => { clearTimeout(timer.current); window.removeEventListener("scroll", hide, true); window.removeEventListener("resize", hide); };
  }, []);
  return <><button {...props} ref={button} type="button" aria-describedby={placement ? id : undefined}
    onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}
    onKeyDown={event => { if (event.key === "Escape") hide(); props.onKeyDown?.(event); }}
    onClick={event => { hide(); props.onClick?.(event); }}>{children}</button>
    {placement && createPortal(<div id={id} role="tooltip" className="history-date-tooltip" style={placement}>
      <strong>{label}</strong>{time && <span>{lang === "zh" ? "创建于" : "Created"} {time}</span>}
    </div>, document.body)}
  </>;
}
