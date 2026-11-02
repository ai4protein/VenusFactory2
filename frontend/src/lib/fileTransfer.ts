/** Browser file transfers only: never turn clipboard text or URLs into uploads. */
export function hasTransferredFiles(data: DataTransfer | null): boolean {
  return !!data && (Array.from(data.types || []).includes('Files') || data.files?.length > 0 || Array.from(data.items || []).some(item => item.kind === 'file'));
}

export function transferredFiles(data: DataTransfer | null): File[] {
  if (!data) return [];
  const files = Array.from(data.files || []);
  if (!files.length) for (const item of Array.from(data.items || [])) {
    if (item.kind !== 'file') continue;
    const file = item.getAsFile();
    if (file) files.push(file);
  }
  return mergeFiles([], files);
}

export function mergeFiles(previous: readonly File[], incoming: readonly File[]): File[] {
  const result = [...previous];
  const key = (file: File) => JSON.stringify([file.name, file.size, file.lastModified, file.type]);
  const seen = new Set(previous.map(key));
  for (const file of incoming) if (!seen.has(key(file))) { seen.add(key(file)); result.push(file); }
  return result;
}

export function proteinFileError(files: readonly File[], zh: boolean): string {
  if (files.length !== 1) return zh ? '请一次选择一个蛋白文件；一个 FASTA 文件可以包含多条序列。' : 'Choose one protein file at a time. A FASTA file can contain multiple sequences.';
  if (!/\.(fa|fasta|faa|pdb|ent|txt)$/i.test(files[0].name)) return zh ? '请选择 FASTA、PDB 或序列 TXT 文件。' : 'Choose a FASTA, PDB or sequence TXT file.';
  return '';
}

export type FileTransferOptions = {
  disabled?: boolean;
  label: string;
  unavailable: string;
  unreadable: string;
  onFiles: (files: File[]) => void | Promise<void>;
  onError: (message: string) => void;
};

/** Scoped to one composer, including portalled controls; returns complete cleanup. */
export function bindFileTransfer(target: HTMLElement, current: () => FileTransferOptions): () => void {
  const clear = () => { target.classList.remove('is-file-dragover'); target.removeAttribute('data-file-drop-label'); };
  const consume = (event: Event, data: DataTransfer | null) => {
    if (!hasTransferredFiles(data)) return;
    event.preventDefault(); event.stopPropagation(); clear();
    const options = current();
    if (options.disabled) { options.onError(options.unavailable); return; }
    const files = transferredFiles(data);
    if (!files.length) { options.onError(options.unreadable); return; }
    try { void Promise.resolve(options.onFiles(files)).catch(error => options.onError(error instanceof Error ? error.message : options.unreadable)); }
    catch (error) { options.onError(error instanceof Error ? error.message : options.unreadable); }
  };
  const paste = (event: ClipboardEvent) => consume(event, event.clipboardData);
  const enter = (event: DragEvent) => {
    if (!hasTransferredFiles(event.dataTransfer)) return;
    event.preventDefault();
    if (!current().disabled) { target.classList.add('is-file-dragover'); target.setAttribute('data-file-drop-label', current().label); }
  };
  const over = (event: DragEvent) => {
    if (!hasTransferredFiles(event.dataTransfer)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = current().disabled ? 'none' : 'copy';
  };
  const leave = (event: DragEvent) => {
    if (event.relatedTarget && target.contains(event.relatedTarget as Node)) return;
    clear();
  };
  const drop = (event: DragEvent) => { consume(event, event.dataTransfer); clear(); };
  const key = (event: KeyboardEvent) => { if (event.key === 'Escape') clear(); };
  const view = target.ownerDocument.defaultView;
  target.addEventListener('paste', paste);
  target.addEventListener('dragenter', enter);
  target.addEventListener('dragover', over);
  target.addEventListener('dragleave', leave);
  target.addEventListener('drop', drop);
  view?.addEventListener('dragend', clear);
  view?.addEventListener('drop', clear);
  view?.addEventListener('blur', clear);
  view?.addEventListener('keydown', key);
  return () => {
    clear();
    target.removeEventListener('paste', paste);
    target.removeEventListener('dragenter', enter);
    target.removeEventListener('dragover', over);
    target.removeEventListener('dragleave', leave);
    target.removeEventListener('drop', drop);
    view?.removeEventListener('dragend', clear);
    view?.removeEventListener('drop', clear);
    view?.removeEventListener('blur', clear);
    view?.removeEventListener('keydown', key);
  };
}
