import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

export function ResearchReportReader({ markdown, identity, zh, composer }: { markdown: string; identity: string; zh: boolean; composer: HTMLElement | null }) {
  const article = useRef<HTMLElement>(null);
  const menu = useRef<HTMLElement>(null);
  const [open, setOpen] = useState(false), [current, setCurrent] = useState(''), [bottom, setBottom] = useState(90);
  const rendered = useMemo(() => {
    const doc = new DOMParser().parseFromString(DOMPurify.sanitize(marked.parse(markdown) as string), 'text/html');
    const headings = Array.from(doc.querySelectorAll('h2')).map((h, i) => {
      // Sequential IDs also cover repeated numbering in the editor's synthesis.
      h.id = `rr-${identity}-section-${i}`;
      return { id: h.id, title: h.textContent || '', index: i };
    });
    return { html: doc.body.innerHTML, headings };
  }, [markdown, identity]);
  useEffect(() => { setOpen(false); setCurrent(''); }, [identity]);
  useEffect(() => {
    const anchor = composer?.closest('.composer') || composer;
    const update = () => setBottom(anchor ? Math.max(20, window.innerHeight - anchor.getBoundingClientRect().top + 12) : 90);
    update(); const observer = new ResizeObserver(update); if (anchor) observer.observe(anchor);
    window.addEventListener('resize', update);
    return () => { observer.disconnect(); window.removeEventListener('resize', update); };
  }, [composer]);
  useEffect(() => {
    const el = article.current; if (!el) return;
    let root = el.parentElement;
    while (root && !/(auto|scroll)/.test(getComputedStyle(root).overflowY)) root = root.parentElement;
    let frame = 0;
    const track = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const top = (root?.getBoundingClientRect().top || 0) + 70;
        const headings = Array.from(el.querySelectorAll('h2[id]'));
        const passed = headings.filter(h => h.getBoundingClientRect().top <= top);
        setCurrent((passed[passed.length - 1] || headings[0])?.id || '');
      });
    };
    track(); document.addEventListener('scroll', track, true); window.addEventListener('resize', track);
    return () => { cancelAnimationFrame(frame); document.removeEventListener('scroll', track, true); window.removeEventListener('resize', track); };
  }, [rendered]);
  useEffect(() => {
    if (!open) return;
    const close = (e: PointerEvent) => { if (!menu.current?.contains(e.target as Node)) setOpen(false); };
    const escape = (e: KeyboardEvent) => { if (e.key === 'Escape') { setOpen(false); menu.current?.querySelector<HTMLButtonElement>('.rr-nav-toggle')?.focus(); } };
    document.addEventListener('pointerdown', close); document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', escape); };
  }, [open]);
  function jump(id?: string) {
    const target = id ? document.getElementById(id) : article.current;
    target?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    setOpen(false);
  }
  const index = rendered.headings.findIndex(h => h.id === current);
  return <>
    <article ref={article} className="rr-report rr-report-reader" dangerouslySetInnerHTML={{ __html: rendered.html }}/>
    {!!rendered.headings.length && createPortal(<nav ref={menu} className={'rr-floating-nav' + (open ? ' is-open' : '')} style={{ bottom }} aria-label={zh ? '报告导航' : 'Report navigation'}>
      {open && <div className="rr-nav-list" id="rr-navigation-list"><header>{zh ? '目录' : 'Contents'}</header>{rendered.headings.map(h => <button type="button" key={h.id} aria-current={current === h.id ? 'location' : undefined} onClick={() => jump(h.id)}><span className="rr-nav-mark"/>{h.title}</button>)}</div>}
      <div className="rr-nav-controls"><button type="button" className="rr-nav-toggle" aria-expanded={open} aria-controls={open ? 'rr-navigation-list' : undefined} onClick={() => setOpen(!open)}><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 5h10M7 10h10M7 15h10M3 5h.01M3 10h.01M3 15h.01"/></svg>{zh ? '目录' : 'Contents'}<small>{Math.max(1, index+1)}/{rendered.headings.length}</small></button><button type="button" onClick={() => jump()} aria-label={zh ? '回到报告开头' : 'Back to report start'} title={zh ? '回到开头' : 'Back to top'}>↑</button></div>
    </nav>, document.body)}
  </>;
}
