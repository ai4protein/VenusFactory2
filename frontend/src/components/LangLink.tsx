import { forwardRef } from "react";
import type { ComponentProps } from "react";
import { Link, Navigate, NavLink } from "react-router-dom";
import { useLang } from "../lib/i18n";

function prefix(lang: string, to: string): string {
  if (!to) return `/${lang}`;
  if (to.startsWith("/")) return `/${lang}${to === "/" ? "" : to}`;
  // Relative paths pass through unchanged (react-router resolves them
  // relative to the current route).
  return to;
}

/** Drop-in replacement for `<Link>` that prefixes an absolute path with the
 *  active language segment, so `<LangLink to="/agent/chat">` becomes
 *  `/en/agent/chat` or `/zh/agent/chat` depending on context. */
export const LangLink = forwardRef<HTMLAnchorElement, ComponentProps<typeof Link>>(
  function LangLink({ to, ...rest }, ref) {
    const { lang } = useLang();
    const target = typeof to === "string" ? prefix(lang, to) : to;
    return <Link ref={ref} to={target} {...rest} />;
  }
);

/** Drop-in replacement for `<NavLink>` with the same prefixing behavior.
 *  `isActive` continues to work because the `to` is the real prefixed URL. */
export const LangNavLink = forwardRef<HTMLAnchorElement, ComponentProps<typeof NavLink>>(
  function LangNavLink({ to, ...rest }, ref) {
    const { lang } = useLang();
    const target = typeof to === "string" ? prefix(lang, to) : to;
    return <NavLink ref={ref} to={target} {...rest} />;
  }
);

/** Drop-in replacement for `<Navigate>` that prefixes an absolute path with
 *  the active language. Use this in route definitions so the redirect target
 *  is always an explicit `/${lang}/...` URL, sidestepping react-router's
 *  relative-resolution rules that depend on where in the tree it's mounted. */
export function LangNavigate({ to, replace }: { to: string; replace?: boolean }) {
  const { lang } = useLang();
  const target = prefix(lang, to);
  return <Navigate to={target} replace={replace} />;
}
