import { useLang } from '../lib/i18n';
import './PageLoading.css';
export function PageLoading() {
  const { lang } = useLang();
  return <div className="page-loading" role="status" aria-label={lang === 'zh' ? '正在加载' : 'Loading'}>
    <div className="page-loading-title"/><div/><div/><div className="page-loading-body"/>
  </div>;
}
