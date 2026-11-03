import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { matchPickerItem, nextPickerItem, pickerPlacement, type PickerItem } from "../lib/pickerNavigation";
import "./ComposerSelect.css";

export function ComposerSelect({ label, ariaLabel = label, value, items, placeholder, disabled = false, disabledHint, helpText, onChange }: {
  label: string; ariaLabel?: string; value: string; items: PickerItem[]; placeholder: string;
  disabled?: boolean; disabledHint?: string; helpText?: string; onChange: (value: string) => void;
}) {
  const id = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const typeahead = useRef({ text: "", at: 0 });
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [placement, setPlacement] = useState<CSSProperties>({});
  const selected = items.find(item => item.value === value);
  const unavailable = disabled || !items.length;

  function show() {
    if (unavailable) return;
    const index = items.findIndex(item => item.value === value && !item.disabled);
    setActive(index < 0 ? nextPickerItem(items, -1, 1) : index);
    typeahead.current = { text: "", at: 0 };
    setOpen(true);
  }
  function choose(index: number) {
    const item = items[index];
    if (!item || item.disabled) return;
    onChange(item.value); setOpen(false); trigger.current?.focus();
  }
  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "Escape") { if (open) { event.preventDefault(); event.stopPropagation(); setOpen(false); } return; }
    if (event.key === "Tab") { setOpen(false); return; }
    if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      if (!open) show();
      else setActive(nextPickerItem(items, event.key === "Home" ? -1 : event.key === "End" ? 0 : active, event.key === "ArrowUp" || event.key === "End" ? -1 : 1));
    } else if (open && ["Enter", " "].includes(event.key)) {
      event.preventDefault(); choose(active);
    } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey && event.key !== " ") {
      event.preventDefault();
      if (!open) show();
      const now = Date.now();
      const text = now - typeahead.current.at < 700 ? typeahead.current.text + event.key : event.key;
      typeahead.current = { text, at: now };
      const index = matchPickerItem(items, text, text.length > 1 ? active - 1 : active);
      if (index >= 0) setActive(index);
    }
  }
  useEffect(() => { if (unavailable) setOpen(false); }, [unavailable]);
  useLayoutEffect(() => {
    if (!open) return;
    const update = () => {
      if (trigger.current) setPlacement(pickerPlacement(trigger.current.getBoundingClientRect(), { width: window.innerWidth, height: window.innerHeight }));
    };
    update();
    const observer = new ResizeObserver(update);
    if (trigger.current) observer.observe(trigger.current);
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    const outside = (event: PointerEvent) => {
      if (!trigger.current?.contains(event.target as Node) && !menu.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => { observer.disconnect(); window.removeEventListener("resize", update); window.removeEventListener("scroll", update, true); document.removeEventListener("pointerdown", outside); };
  }, [open]);
  useEffect(() => { if (open) menu.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" }); }, [open, active]);

  return <>
    <span className={helpText ? "composer-picker-wrap has-help" : "composer-picker-wrap"}>
    <button ref={trigger} type="button" className="composer-picker" role="combobox" aria-label={ariaLabel}
      aria-expanded={open} aria-haspopup="listbox" aria-controls={open ? id : undefined}
      aria-activedescendant={open && active >= 0 ? `${id}-${active}` : undefined} disabled={unavailable}
      title={disabledHint || selected?.label || placeholder} onClick={() => open ? setOpen(false) : show()}
      onKeyDown={onKeyDown} onBlur={event => { if (!menu.current?.contains(event.relatedTarget)) setOpen(false); }}>
      <span>{selected?.label || placeholder}</span>
      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m5 6 3 3 3-3"/></svg>
    </button>
    {helpText && <button type="button" className="composer-picker-help" aria-label={helpText} title={helpText}>?</button>}
    </span>
    {open && createPortal(<div ref={menu} className="composer-picker-menu" style={placement}>
      <div className="composer-picker-heading">{label}</div>
      <div id={id} role="listbox" aria-label={ariaLabel} className="composer-picker-options">
        {items.map((item, index) => <div key={item.value} id={`${id}-${index}`} role="option"
          aria-selected={item.value === value} aria-disabled={item.disabled || undefined} data-index={index}
          className={`composer-picker-option${active === index ? " is-active" : ""}`}
          onMouseEnter={() => { if (!item.disabled) setActive(index); }} onMouseDown={event => event.preventDefault()} onClick={() => choose(index)}>
          <span>{item.label}{item.disabled && item.hint && <small>{item.hint}</small>}</span>
          {item.value === value && <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m3 8 3 3 7-7"/></svg>}
        </div>)}
      </div>
    </div>, document.body)}
  </>;
}
