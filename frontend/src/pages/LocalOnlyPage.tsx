import { PageFooter } from "../components/PageFooter";

export function LocalOnlyPage({ title }: { title: string }) {
  return (
    <div className="module-shell">
      <h2>{title}</h2>
      <p>This module is shown in online mode but disabled.</p>
      <div className="module-shell-card local-only-card">
        <h3>Local Only</h3>
        <p>This module is available only in local mode. Start the app in local mode to enable full functionality.</p>
      </div>
      <PageFooter />
    </div>
  );
}
