import { useEffect } from "react";
import { useLocation } from "react-router-dom";

interface Meta {
  title: string;
  description?: string;
}

function setOrCreateMeta(name: string, content: string) {
  let tag = document.querySelector(`meta[name="${name}"]`) as HTMLMetaElement | null;
  let created = false;
  if (!tag) {
    tag = document.createElement("meta");
    tag.setAttribute("name", name);
    document.head.appendChild(tag);
    created = true;
  }
  const previous = tag.getAttribute("content");
  tag.setAttribute("content", content);
  return () => {
    if (created) {
      tag?.remove();
    } else if (previous !== null) {
      tag?.setAttribute("content", previous);
    }
  };
}

function setOrCreateLink(rel: string, href: string, hreflang?: string) {
  const selector = hreflang
    ? `link[rel="${rel}"][hreflang="${hreflang}"]`
    : `link[rel="${rel}"]:not([hreflang])`;
  let tag = document.querySelector(selector) as HTMLLinkElement | null;
  let created = false;
  if (!tag) {
    tag = document.createElement("link");
    tag.setAttribute("rel", rel);
    if (hreflang) tag.setAttribute("hreflang", hreflang);
    document.head.appendChild(tag);
    created = true;
  }
  const previous = tag.getAttribute("href");
  tag.setAttribute("href", href);
  return () => {
    if (created) {
      tag?.remove();
    } else if (previous !== null) {
      tag?.setAttribute("href", previous);
    }
  };
}

/** Build absolute URL for the current path, swapping `/${currentLang}/` →
 *  `/${targetLang}/` to produce per-language canonical / hreflang siblings. */
function buildLangUrl(currentPath: string, currentLang: string, targetLang: string): string {
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  if (!currentLang) {
    return `${origin}/${targetLang}${currentPath === "/" ? "" : currentPath}`;
  }
  const swapped = currentPath.replace(new RegExp(`^/${currentLang}(?=/|$)`), `/${targetLang}`);
  return origin + swapped;
}

/** Updates <title>, <meta name="description">, <link rel="canonical">, and
 *  <link rel="alternate" hreflang="..."> at runtime. SPA routes don't get
 *  unique titles or per-locale canonical URLs automatically; call this from
 *  a page so search engines see the right metadata per route AND per lang. */
export function useDocumentMeta({ title, description }: Meta) {
  const { pathname } = useLocation();
  useEffect(() => {
    if (typeof document === "undefined") return;

    const restorers: Array<() => void> = [];

    const previousTitle = document.title;
    document.title = title;
    restorers.push(() => {
      document.title = previousTitle;
    });

    if (description !== undefined) {
      restorers.push(setOrCreateMeta("description", description));
    }

    // Determine the active lang from URL: /en/* or /zh/*
    const seg = pathname.split("/").filter(Boolean)[0];
    const lang = seg === "en" || seg === "zh" ? seg : "";

    // Canonical URL: the lang-prefixed version of the current path.
    const enUrl = buildLangUrl(pathname, lang, "en");
    const zhUrl = buildLangUrl(pathname, lang, "zh");
    const selfUrl = lang === "zh" ? zhUrl : enUrl;

    restorers.push(setOrCreateLink("canonical", selfUrl));
    restorers.push(setOrCreateLink("alternate", enUrl, "en"));
    restorers.push(setOrCreateLink("alternate", zhUrl, "zh-CN"));

    return () => {
      // Run restorers in reverse order.
      for (let i = restorers.length - 1; i >= 0; i--) restorers[i]();
    };
  }, [title, description, pathname]);
}
