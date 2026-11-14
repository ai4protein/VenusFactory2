import type { Dataset } from './modelPlatform';

const aminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
export function sequenceColumn(data: Pick<Dataset, 'columns' | 'preview'>): string {
  const named = data.columns.find(c => /^(sequence|seq|aa_seq|protein_sequence|amino_acid_sequence)$/i.test(c));
  if (named) return named;
  const candidates = data.columns.filter(c => data.preview.length > 0 && data.preview.every(r => {
    const v = String(r[c] ?? '').trim();
    return v.length >= 5 && aminoAcids.test(v);
  }));
  return candidates.length === 1 ? candidates[0] : '';
}

export function numericColumns(data: Pick<Dataset, 'columns' | 'preview'>): string[] {
  return data.columns.filter(c => data.preview.length > 0 && data.preview.every(r =>
    r[c] !== null && r[c] !== undefined && String(r[c]).trim() !== '' && Number.isFinite(Number(r[c]))));
}

/** A wrapped raw sequence is one protein; FASTA headers delimit multiple proteins. */
export function parseProteinInput(text: string, column = 'sequence', zh = false): Record<string, string>[] {
  const fail = (en: string, cn: string): never => { throw new Error(zh ? cn : en); };
  const lines = text.trim().split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  if (!lines.length) fail('Paste a protein sequence or FASTA.', '请粘贴蛋白序列或 FASTA。');
  const records: { name: string; sequence: string }[] = [];
  if (lines[0].startsWith('>')) {
    for (const line of lines) {
      if (line.startsWith('>')) records.push({ name: line.slice(1).trim() || `protein_${records.length + 1}`, sequence: '' });
      else records[records.length - 1].sequence += line.replace(/\s/g, '');
    }
  } else records.push({ name: 'protein_1', sequence: lines.join('').replace(/\s/g, '') });
  if (records.length > 10000) fail('Use up to 10,000 proteins per run.', '每次最多输入 10,000 条蛋白序列。');
  for (const r of records) {
    if (!r.sequence) fail(`Missing sequence: ${r.name}`, `缺少序列：${r.name}`);
    if (!aminoAcids.test(r.sequence)) fail(`Check ${r.name}: use the 20 standard amino acids.`, `请检查 ${r.name}：使用 20 种标准氨基酸字母。`);
  }
  const idColumn = column === 'name' ? 'record_name' : 'name';
  return records.map(r => ({ [idColumn]: r.name, [column]: r.sequence.toUpperCase() }));
}

export type TableResult = { rows?: Record<string, unknown>[]; outputs?: Record<string, { rows?: Record<string, unknown>[] }> };
export function resultRows(result: TableResult): Record<string, unknown>[] {
  if (result.rows) return result.rows;
  const tables = Object.entries(result.outputs || {}).filter(([, value]) => value.rows?.length);
  if (tables.length === 1) return tables[0][1].rows!;
  const used = new Set(tables.flatMap(([, v]) => v.rows!.flatMap(r => Object.keys(r))));
  let key = 'result_output'; while (used.has(key)) key = '_' + key;
  return tables.flatMap(([name, value]) => value.rows!.map(row => ({ [key]: name, ...row })));
}
export function tableCsv(rows: Record<string, unknown>[]): string {
  const columns = Array.from(new Set(rows.flatMap(row => Object.keys(row))));
  const cell = (value: unknown) => {
    let s = typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '');
    // Keep text values literal when opened in a spreadsheet; numeric values stay numeric.
    if (typeof value === 'string' && /^[\s]*[=+\-@]/.test(s)) s = "'" + s;
    return '"' + s.replace(/"/g, '""') + '"';
  };
  return '\uFEFF' + [columns.map(cell).join(','), ...rows.map(row => columns.map(c => cell(row[c])).join(','))].join('\r\n');
}
