export type RsiCase = { id: string; title: string; summary: string; author_email?: string; visibility: 'private' | 'public'; status: string; created_at?: string | number; session_id?: string };
export type RsiFile = { path: string; size: number };
export type RsiCaseDetail = RsiCase & { files: RsiFile[]; readme?: string; report?: string; download_url: string };
export type RsiCatalog = { items: RsiCase[]; total: number; next_offset: number | null };
export type RsiPreview = { path: string; text: string; truncated: boolean; size: number; binary?: boolean };
export type RsiTree = { name: string; path: string; folders: RsiTree[]; files: RsiFile[] };

export function caseSession(id: string): string {
  return /^research:([a-f0-9-]+):[a-f0-9]+$/i.exec(id)?.[1] || '';
}
export function caseEndpoint(id: string, suffix = ''): string {
  return '/api/rsi-plaza/cases/' + encodeURIComponent(id) + suffix;
}
export function repositoryTree(files: RsiFile[]): RsiTree {
  const root: RsiTree = { name: '', path: '', folders: [], files: [] };
  for (const file of files) {
    const parts = file.path.split('/');
    if (parts.some(part => !part || part === '.' || part === '..')) continue;
    let node = root;
    for (const name of parts.slice(0, -1)) {
      let folder = node.folders.find(child => child.name === name);
      if (!folder) { folder = { name, path: node.path + name + '/', folders: [], files: [] }; node.folders.push(folder); }
      node = folder;
    }
    node.files.push(file);
  }
  const sort = (node: RsiTree) => { node.folders.sort((a,b) => a.name.localeCompare(b.name)); node.files.sort((a,b) => a.path.localeCompare(b.path)); node.folders.forEach(sort); };
  sort(root); return root;
}
export function firstCaseFile(detail: Pick<RsiCaseDetail, 'files' | 'readme' | 'report'>): string {
  const paths = new Set(detail.files.map(file => file.path));
  if (detail.readme && paths.has(detail.readme)) return detail.readme;
  return detail.files.find(file => /^readme\.(md|txt)$/i.test(file.path))?.path || detail.files.find(file => /(^|\/)readme\.(md|txt)$/i.test(file.path))?.path || detail.files[0]?.path || '';
}
export function caseDate(value: string | number | undefined, language: string): string {
  if (!value) return '';
  const date = new Date(typeof value === 'number' ? value * 1000 : value);
  return Number.isFinite(date.getTime()) ? date.toLocaleDateString(language === 'zh' ? 'zh-CN' : 'en-GB') : '';
}
