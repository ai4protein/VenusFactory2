/** Standard Google tag; GA enhanced measurement owns SPA history page views. */
export type AnalyticsRuntimeConfig = {
  mode?: string;
  google_analytics_id?: string;
};

type AnalyticsWindow = Pick<Window, "document"> & {
  dataLayer?: unknown[];
  gtag?: (...args: unknown[]) => void;
};

const SCRIPT_ID = "venus-google-analytics";

export function initializeGoogleAnalytics(
  config: AnalyticsRuntimeConfig,
  target: AnalyticsWindow = window,
): void {
  const id = config.google_analytics_id?.trim() ?? "";
  if (config.mode !== "online" || !/^G-[A-Z0-9]{4,32}$/.test(id)) return;
  // StrictMode, language changes and shell remounts must not configure GA twice.
  if (target.document.getElementById(SCRIPT_ID)) return;

  target.dataLayer = target.dataLayer || [];
  target.gtag = target.gtag || function (..._args: unknown[]) {
    target.dataLayer!.push(arguments);
  };
  target.gtag("js", new Date());
  target.gtag("config", id);

  const script = target.document.createElement("script");
  script.id = SCRIPT_ID;
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${id}`;
  script.referrerPolicy = "strict-origin-when-cross-origin";
  target.document.head.appendChild(script);
}
