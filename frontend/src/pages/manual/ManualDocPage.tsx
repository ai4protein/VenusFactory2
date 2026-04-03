import { useEffect, useMemo, useRef, useState } from "react";
import { useOutletContext, useParams } from "react-router-dom";
import {
  getManualSection,
  loadManualMarkdown,
  renderManualHtml,
  type TocItem
} from "../../lib/manualContent";
import type { ManualLayoutContext } from "./ManualLayout";

function buildNotFoundHtml() {
  return "<h1>Manual section not found</h1><p>Please choose a valid section from the tabs.</p>";
}

export function ManualDocPage() {
  const { section: sectionParam } = useParams<{ section: string }>();
  const section = useMemo(() => getManualSection(sectionParam), [sectionParam]);
  const { language } = useOutletContext<ManualLayoutContext>();

  const [loading, setLoading] = useState(true);
  const [toc, setToc] = useState<TocItem[]>([]);
  const [html, setHtml] = useState("");
  const [activeId, setActiveId] = useState("");
  const [modalImage, setModalImage] = useState<{ src: string; alt: string } | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let alive = true;
    async function run() {
      setLoading(true);
      setActiveId("");
      if (!section) {
        setToc([]);
        setHtml(buildNotFoundHtml());
        setLoading(false);
        return;
      }
      const markdown = await loadManualMarkdown(section.key, language);
      if (!alive) return;
      const rendered = renderManualHtml(markdown);
      setToc(rendered.toc);
      setHtml(rendered.html);
      setLoading(false);
    }
    void run();
    return () => {
      alive = false;
    };
  }, [language, section]);

  useEffect(() => {
    if (!contentRef.current) return;
    const root = contentRef.current;

    const handleImageClick = (event: Event) => {
      const target = event.target as HTMLElement | null;
      if (!target || target.tagName.toLowerCase() !== "img") return;
      const image = target as HTMLImageElement;
      setModalImage({ src: image.src, alt: image.alt || "Manual image" });
    };
    root.addEventListener("click", handleImageClick);

    const images = Array.from(root.querySelectorAll("img"));
    const restoreFns: Array<() => void> = [];
    for (const image of images) {
      const onError = () => {
        const holder = document.createElement("div");
        holder.className = "manual-v2-image-error";
        holder.textContent = `[Image does not exist: ${image.getAttribute("src") || "unknown"}]`;
        image.replaceWith(holder);
      };
      image.addEventListener("error", onError, { once: true });
      restoreFns.push(() => image.removeEventListener("error", onError));
    }

    return () => {
      root.removeEventListener("click", handleImageClick);
      for (const fn of restoreFns) fn();
    };
  }, [html]);

  useEffect(() => {
    function syncActiveHeading() {
      if (!contentRef.current || toc.length === 0) return;
      const topOffset = 140;
      let currentId = toc[0].id;
      for (const item of toc) {
        const anchor = contentRef.current.querySelector(`#${item.id}`);
        if (!anchor) continue;
        const top = anchor.getBoundingClientRect().top;
        if (top <= topOffset) {
          currentId = item.id;
        } else {
          break;
        }
      }
      setActiveId(currentId);
    }

    syncActiveHeading();
    window.addEventListener("scroll", syncActiveHeading, { passive: true });
    return () => window.removeEventListener("scroll", syncActiveHeading);
  }, [toc, html]);

  return (
    <section className="manual-v2-panel">
      <aside className="manual-v2-nav">
        {loading ? (
          <p>{language === "Chinese" ? "目录加载中..." : "Loading outline..."}</p>
        ) : toc.length === 0 ? (
          <p>{language === "Chinese" ? "未找到标题。" : "No headings found."}</p>
        ) : (
          <ul>
            {toc.map((item) => (
              <li key={item.id}>
                <a
                  href={`#${item.id}`}
                  className={`manual-v2-nav-link level-${item.level} ${activeId === item.id ? "active" : ""}`}
                >
                  {item.title}
                </a>
              </li>
            ))}
          </ul>
        )}
      </aside>

      <div className="manual-v2-content-wrap">
        <article
          ref={contentRef}
          className="manual-v2-content manual-v2-markdown"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>

      {modalImage && (
        <div className="manual-v2-image-modal" onClick={() => setModalImage(null)}>
          <img src={modalImage.src} alt={modalImage.alt} />
        </div>
      )}
    </section>
  );
}
