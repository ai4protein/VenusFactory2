import { PageFooter } from "../components/PageFooter";

export function ModuleShellPage({ title, status }: { title: string; status: string }) {
  return (
    <div className="module-shell">
      <h2>{title}</h2>
      <p>{status}</p>
      <div className="module-shell-card">
        <h3>Migration Placeholder</h3>
        <p>
          This module is registered in the new navigation and ready for incremental migration.
          Existing functionality is still available in the legacy Gradio entry.
        </p>
      </div>
      <PageFooter />
    </div>
  );
}
