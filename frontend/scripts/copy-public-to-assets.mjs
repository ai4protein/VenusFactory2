import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";

const publicDir = "public";
const destDir = "dist/assets";
mkdirSync(destDir, { recursive: true });

for (const name of readdirSync(publicDir)) {
  if (name === "assets") continue;
  copyFileSync(join(publicDir, name), join(destDir, name));
}
