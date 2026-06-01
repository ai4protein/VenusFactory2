import { createContext, useCallback, useContext, useEffect, useMemo } from "react";
import type { ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";

export type Lang = "en" | "zh";

const STORAGE_KEY = "vf2:lang";
const LANG_VALUES = new Set<Lang>(["en", "zh"]);
export const DEFAULT_LANG: Lang = "en";

/** Per-component translation dictionary.
 *  T is intentionally unconstrained: pages may use flat string maps, nested
 *  groups (`{ nav: { ... } }`), or readonly `as const` literals. */
export type LangDict<T> = { en: T; zh: T };

interface LanguageContextValue {
  lang: Lang;
  setLang: (next: Lang) => void;
  t: <T>(dict: LangDict<T>) => T;
}

const LanguageContext = createContext<LanguageContextValue | null>(null);

/** Extract the `:lang` segment from a pathname like `/en/foo/bar`.
 *  Returns null when the first segment is not a known lang code. */
export function langFromPath(pathname: string): Lang | null {
  const seg = pathname.split("/").filter(Boolean)[0];
  return LANG_VALUES.has(seg as Lang) ? (seg as Lang) : null;
}

/** Replace the leading `:lang` segment of `pathname` with `nextLang`.
 *  If pathname has no lang prefix, prepends one. Preserves search + hash. */
export function withLangPath(pathname: string, nextLang: Lang): string {
  const segments = pathname.split("/").filter(Boolean);
  if (segments.length > 0 && LANG_VALUES.has(segments[0] as Lang)) {
    segments[0] = nextLang;
  } else {
    segments.unshift(nextLang);
  }
  return "/" + segments.join("/");
}

/** Compute the language to redirect bare paths to. Precedence:
 *   1. `?lang=` query param
 *   2. localStorage saved preference
 *   3. English default (English-first product) */
export function readDefaultLang(): Lang {
  try {
    if (typeof window !== "undefined") {
      const fromQuery = new URLSearchParams(window.location.search).get("lang");
      if (fromQuery === "en" || fromQuery === "zh") return fromQuery;
    }
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "zh") return saved;
  } catch {
    /* ignore */
  }
  return DEFAULT_LANG;
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();

  // Source of truth is the URL. Falls back to default if the URL is bare
  // (a redirect from `/` to `/${default}` happens via routes).
  const lang = langFromPath(location.pathname) ?? readDefaultLang();

  // Remember chosen language so the next bare-path entry redirects to it.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      /* ignore */
    }
    if (typeof document !== "undefined") {
      document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    }
  }, [lang]);

  const setLang = useCallback(
    (next: Lang) => {
      if (next === lang) return;
      const target = withLangPath(location.pathname, next) + location.search + location.hash;
      navigate(target, { replace: false });
    },
    [lang, location.pathname, location.search, location.hash, navigate]
  );

  const value = useMemo<LanguageContextValue>(
    () => ({
      lang,
      setLang,
      t: (dict) => dict[lang]
    }),
    [lang, setLang]
  );

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLang(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLang must be used inside <LanguageProvider>");
  }
  return ctx;
}
