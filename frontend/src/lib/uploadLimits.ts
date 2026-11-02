export const ONLINE_UPLOAD_BYTES = 10 * 1024 * 1024;
type UploadFileSize = Pick<File, "name" | "size">;

export function uploadSizeError(files: readonly UploadFileSize[], mode: string | undefined, lang?: string): string {
  if (mode !== "online") return "";
  const tooLarge = files.find(file => file.size > ONLINE_UPLOAD_BYTES);
  if (!tooLarge) return "";
  const zh = (lang || window.location.pathname.split("/")[1]) === "zh";
  return zh ? `${tooLarge.name} 超过 10 MB，请选择更小的文件。` : `${tooLarge.name} exceeds 10 MB. Choose a smaller file.`;
}

let runtime: Promise<{ mode: string }> | undefined;
async function runtimeMode() {
  if (!runtime) runtime = fetch("/api/runtime-config", { cache: "no-store" }).then(async response => {
    if (!response.ok) throw new Error("Unable to check upload settings. Please try again.");
    const config = await response.json() as { mode: string };
    if (!["online", "local"].includes(config.mode)) throw new Error("Upload settings are not ready. Please try again.");
    return config;
  }).finally(() => { runtime = undefined; });
  return (await runtime).mode;
}

export async function validateUploads(files: readonly UploadFileSize[]) {
  const error = uploadSizeError(files, await runtimeMode());
  if (error) throw new Error(error);
}
