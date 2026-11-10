/** CSV/TSV display parsing. Keep identifiers, precision and quoted text verbatim. */
export type DelimitedTable = { headers: string[]; rows: string[][]; truncated: boolean };
type Limits = { maxRows?: number; maxChars?: number };

export function parseDelimitedTable(text: string, separator: "," | "\t" = ",", limits: Limits = {}): DelimitedTable {
  const maxRows = limits.maxRows ?? Infinity;
  const maxChars = limits.maxChars ?? text.length;
  if ((maxRows !== Infinity && (!Number.isSafeInteger(maxRows) || maxRows < 0)) ||
      !Number.isSafeInteger(maxChars) || maxChars < 0) throw new Error("Invalid table preview limit");
  const end = Math.min(text.length, maxChars);
  let headers: string[] | null = null;
  const rows: string[][] = [];
  let row: string[] = [], cell = "", quoted = false, closedQuote = false, started = false;
  const finish = () => {
    if (!started && row.length === 0 && cell === "") return;
    row.push(cell);
    if (headers === null) headers = row;
    else {
      if (row.length !== headers.length) throw new Error("Table records have different column counts");
      rows.push(row);
    }
    row = []; cell = ""; closedQuote = false; started = false;
  };
  for (let i = text.startsWith("\uFEFF") ? 1 : 0; i < end; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (i + 1 < end && text[i + 1] === '"') { cell += '"'; i += 1; }
        else { quoted = false; closedQuote = true; }
      } else cell += ch;
      continue;
    }
    if (ch === separator) {
      row.push(cell); cell = ""; closedQuote = false; started = true;
    } else if (ch === "\n" || ch === "\r") {
      finish();
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      if (headers !== null && rows.length >= maxRows) {
        return { headers, rows, truncated: i + 1 < text.length };
      }
    } else if (closedQuote) {
      if (ch !== " ") throw new Error("Unexpected text after a quoted table cell");
    } else if (ch === '"') {
      if (cell !== "") throw new Error("Unexpected quote in a table cell");
      quoted = true; started = true;
    } else {
      cell += ch; started = true;
    }
  }
  if (end < text.length) return { headers: headers ?? [], rows, truncated: true };
  if (quoted) throw new Error("Unclosed quoted table cell");
  finish();
  return { headers: headers ?? [], rows, truncated: false };
}

/** Object-based result views need distinct keys, including for duplicate headers. */
export function tableRecords(table: DelimitedTable): Record<string, string>[] {
  const reserved = new Set(table.headers.filter(Boolean));
  const used = new Set<string>();
  const keys = table.headers.map((header, i) => {
    const base = header || `column_${i + 1}`;
    let key = base, suffix = 2;
    while (used.has(key) || (!header && reserved.has(key)) || (key !== base && reserved.has(key))) {
      key = `${base} (${suffix++})`;
    }
    used.add(key);
    return key;
  });
  return table.rows.map(row => Object.fromEntries(keys.map((key, i) => [key, row[i]])));
}
