export const STRUCTURE_EXTENSIONS = new Set(["pdb", "ent", "cif", "mmcif", "bcif", "mol", "sdf", "mol2"]);
export function structureExtension(path: string): string {
  let name = path;
  try { const url = new URL(path, "http://venus.local"); name = url.searchParams.get("path") || url.pathname; } catch { /* plain filename */ }
  try { name = decodeURIComponent(name); } catch { /* retain original */ }
  return name.split(/[?#]/)[0].split(".").pop()?.toLowerCase() || "";
}
export function isStructureFile(path: string): boolean { return STRUCTURE_EXTENSIONS.has(structureExtension(path)); }
export function structureFormat(ext: string): string { return ({ ent: "pdb", cif: "mmcif", bcif: "mmcif" } as Record<string, string>)[ext.toLowerCase()] || ext.toLowerCase(); }
export type StructureRequest = { path: string; name?: string; url?: string; ext?: string };
export function openStructure(request: StructureRequest) { window.dispatchEvent(new CustomEvent("venus:open-structure", { detail: request })); }
export function structurePathFromLink(href: string): string {
  try { return new URL(href, window.location.origin).searchParams.get("path") || href; } catch { return href; }
}
export function matchSessionFile<T extends { key: string; rel: string; name: string }>(files: T[], path: string): T | undefined {
  const clean = path.replace(/\\/g, "/").replace(/[?#].*$/, "");
  const exact = files.filter(f => f.key === clean || f.rel === clean || (f.rel.includes("/") && clean.endsWith("/" + f.rel)) || clean.endsWith("/" + f.key));
  if (exact.length) return exact.length === 1 ? exact[0] : undefined;
  const name = clean.split("/").pop();
  const matches = files.filter(f => f.name === name);
  return matches.length === 1 ? matches[0] : undefined;
}
