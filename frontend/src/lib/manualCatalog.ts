import catalog from '../../../docs/manual/catalog.json' with { type: 'json' };
import type { Lang } from './i18n';

export type ManualLanguage = 'English' | 'Chinese';
export type ManualSectionKey = string;
export const MANUAL_SECTIONS = catalog.sections.map(item => ({
  ...item, label: item.title.en,
  files: { English: item.files.en, Chinese: item.files.zh },
}));
export function langToManual(lang: Lang): ManualLanguage { return lang === 'zh' ? 'Chinese' : 'English'; }
export function getManualSection(key?: string) {
  return MANUAL_SECTIONS.find(item => item.key === key || item.aliases.includes(key || '')) ?? null;
}

/** Recover old links that accumulated valid chapter names during navigation. */
export function canonicalManualPath(pathname: string, lang: Lang): string | null {
  const base = `/${lang}/manual`;
  if (pathname !== base && !pathname.startsWith(`${base}/`)) return null;
  const parts = pathname.slice(base.length).split('/').filter(Boolean);
  const sections = parts.map(part => part === 'index' ? 'index' : getManualSection(part)?.key);
  if (sections.some(section => !section)) return null;
  return `${base}/${sections.at(-1) ?? 'index'}`;
}
export function manualQuestion(key: string | null, lang: Lang): string {
  const section = key ? getManualSection(key) : null;
  if (!section) return '';
  return lang === 'zh'
    ? `请参考平台当前手册，介绍「${section.title.zh}」的使用步骤、所需输入和相关限制，并附上对应教程链接。`
    : `Using the current platform manual, explain how to use ${section.title.en}, including the steps, required inputs and relevant limits. Link to the tutorial.`;
}
export function manualChatHref(key: string, lang: Lang) {
  return `/${lang}/agent/chat?mode=agent&help=${encodeURIComponent(key)}`;
}
