import { ReactNode } from "react";
import { PageFooter } from "../../components/PageFooter";

type QuickToolsLayoutProps = {
  title: string;
  subtitle: string;
  running: boolean;
  progress?: number;
  progressMessage?: string;
  left: ReactNode;
  right: ReactNode;
};

export function QuickToolsLayout(props: QuickToolsLayoutProps) {
  return (
    <div className="quick-tools-v2-page">
      <header className="chat-header">
        <div>
          <h2>{props.title}</h2>
          <p>{props.subtitle}</p>
        </div>
        <div className={`run-status-bar ${props.running ? "running" : "stopped"}`}>
          <span className="run-status-dot" />
          <span className="run-status-text">{props.running ? "Task Running" : "Ready"}</span>
        </div>
      </header>

      <section className="quick-tools-v2-grid">
        <aside className="chat-panel left quick-tools-v2-left">{props.left}</aside>
        <section className="chat-panel center quick-tools-v2-right">{props.right}</section>
      </section>
      {props.running && (
        <section className="chat-panel custom-section-card">
          <h3>Progress</h3>
          <div className="custom-progress-wrap">
            <div className="custom-progress-meta">
              <span>{props.progressMessage || "Running..."}</span>
              <span>{Math.round(Math.max(0, Math.min(1, props.progress ?? 0)) * 100)}%</span>
            </div>
            <div
              className="custom-progress-track"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(Math.max(0, Math.min(1, props.progress ?? 0)) * 100)}
            >
              <div
                className="custom-progress-fill"
                style={{ width: `${Math.round(Math.max(0, Math.min(1, props.progress ?? 0)) * 100)}%` }}
              />
            </div>
          </div>
        </section>
      )}
      <PageFooter />
    </div>
  );
}
