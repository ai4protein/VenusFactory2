import { useLang } from '../lib/i18n';
import { ModelGlyph } from '../components/ModelPlatformShared';
import '../online-library.css';

export function OnlineLibraryPage() {
  const {lang} = useLang(); const zh = lang === 'zh';
  return <main className="online-library-unavailable"><h1>{zh ? '文件库' : 'Library'}</h1><div role="status"><ModelGlyph kind="data" size={32}/><p>{zh ? 'Online 模式不可用' : 'Unavailable in online mode'}</p></div></main>;
}
