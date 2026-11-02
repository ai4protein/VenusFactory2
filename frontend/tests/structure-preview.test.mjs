import test from "node:test";
import assert from "node:assert/strict";
import { isStructureFile, structureExtension, structureFormat, matchSessionFile } from "../src/lib/structurePreview.ts";
test("protein and ligand formats, URLs and scoped API links", () => {
  for (const name of ["x.PDB", "ligand.sdf", "a.mol2", "a.ent", "a.mmcif", "a.bcif", "https://example.org/p.cif?v=1", "/api/content?path=uploads%2Fa.pdb"]) assert.ok(isStructureFile(name));
  assert.equal(isStructureFile("protein.fasta"), false); assert.equal(isStructureFile("model.pdb.exe"), false);
  assert.equal(structureExtension("/api/content?path=uploads%2Fa.pdb"), "pdb");
  assert.equal(structureFormat("ent"), "pdb"); assert.equal(structureFormat("cif"), "mmcif"); assert.equal(structureFormat("mol2"), "mol2");
});
test("file resolution preserves session scope and refuses ambiguous basenames", () => {
  const files = [{ key: "uploads/a.pdb", rel: "a.pdb", name: "a.pdb" }, { key: "work/a.pdb", rel: "out/a.pdb", name: "a.pdb" }, { key: "work/b.sdf", rel: "b.sdf", name: "b.sdf" }];
  assert.equal(matchSessionFile(files, "uploads/a.pdb"), files[0]);
  assert.equal(matchSessionFile(files, "/workspace/out/a.pdb"), files[1]);
  assert.equal(matchSessionFile(files, "/unknown/a.pdb"), undefined);
  assert.equal(matchSessionFile(files, "/workspace/b.sdf"), files[2]);
  assert.equal(matchSessionFile(files, "/other/missing.pdb"), undefined);
});
