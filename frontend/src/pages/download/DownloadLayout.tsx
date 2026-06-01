import { ReactNode } from "react";
import { PageFooter } from "../../components/PageFooter";
import { useLang } from "../../lib/i18n";

type DownloadLayoutProps = {
  title: string;
  subtitle: string;
  running: boolean;
  left: ReactNode;
  right: ReactNode;
};

const STRINGS = {
  en: { running: "Downloading", idle: "Ready" },
  zh: { running: "下载中", idle: "就绪" }
};

export function DownloadLayout(props: DownloadLayoutProps) {
  const t = useLang().t(STRINGS);
  return (
    <div className="download-v2-page">
      <header className="chat-header">
        <div>
          <h2>{props.title}</h2>
          <p>{props.subtitle}</p>
        </div>
        <div className={`run-status-bar ${props.running ? "running" : "stopped"}`}>
          <span className="run-status-dot" />
          <span className="run-status-text">{props.running ? t.running : t.idle}</span>
        </div>
      </header>

      <section className="download-v2-grid">
        <aside className="chat-panel left download-v2-left">{props.left}</aside>
        <section className="chat-panel center download-v2-right">{props.right}</section>
      </section>
      <PageFooter />
    </div>
  );
}
