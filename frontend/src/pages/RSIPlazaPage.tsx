import { useEffect, useMemo, useState } from 'react';
import DOMPurify from 'dompurify';
import { useSearchParams } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import { quotaHeaders } from '../lib/onlineQuota';
import { getChatSessionAuthHeaders } from '../lib/api';
import { renderMarkdown } from '../lib/markdown';
import { caseDate, caseEndpoint, caseSession, firstCaseFile, repositoryTree, type RsiCaseDetail, type RsiCatalog, type RsiPreview, type RsiTree } from '../lib/rsiPlaza';
import './RSIPlazaPage.css';

function renderCaseMarkdown(text: string): string {
  return DOMPurify.sanitize(renderMarkdown(text), {
    ALLOWED_TAGS: ['p', 'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote',
      'strong', 'em', 'b', 'i', 's', 'del', 'sub', 'sup', 'ul', 'ol', 'li', 'dl', 'dt', 'dd',
      'pre', 'code', 'kbd', 'samp', 'table', 'thead', 'tbody', 'tfoot', 'tr', 'th', 'td',
      'details', 'summary', 'a', 'span', 'div'],
    ALLOWED_ATTR: ['href', 'title', 'colspan', 'rowspan', 'start', 'open'],
    ALLOW_DATA_ATTR: false, ALLOW_ARIA_ATTR: false,
  });
}
async function read<T>(url: string, signal: AbortSignal, id = ''): Promise<T> {
  const response = await fetch(url, { signal, credentials: 'same-origin', cache: 'no-store', headers: id ? getChatSessionAuthHeaders(caseSession(id)) : quotaHeaders() });
  if (!response.ok) throw new Error(String(response.status));
  return response.json() as Promise<T>;
}
function FileTree({ node, selected, choose }: { node: RsiTree; selected: string; choose: (path: string) => void }) {
  return <ul>{node.folders.map(folder => <li key={folder.path}><details open><summary title={folder.path}>{folder.name}</summary><FileTree node={folder} selected={selected} choose={choose}/></details></li>)}{node.files.map(file => <li key={file.path}><button type="button" title={file.path} aria-current={selected === file.path ? 'true' : undefined} onClick={() => choose(file.path)}><span aria-hidden="true">▤</span>{file.path.split('/').pop()}</button></li>)}</ul>;
}
export function RSIPlazaPage() {
  const { lang } = useLang(), zh = lang === 'zh';
  const say = (en: string, cn: string) => zh ? cn : en;
  const [params, setParams] = useSearchParams();
  const id = params.get('case') || '', mine = params.get('view') === 'mine';
  const [query, setQuery] = useState(''), [offset, setOffset] = useState(0), [revision, setRevision] = useState(0);
  const [catalog, setCatalog] = useState<RsiCatalog>(), [loading, setLoading] = useState(false), [error, setError] = useState('');
  const [detail, setDetail] = useState<RsiCaseDetail>(), [detailError, setDetailError] = useState('');
  const [selected, setSelected] = useState(''), [fileQuery, setFileQuery] = useState('');
  const [preview, setPreview] = useState<{ key: string; value?: RsiPreview; error?: string }>();
  const [downloading, setDownloading] = useState(false), [downloadError, setDownloadError] = useState('');
  const chosen = detail?.id === id ? (detail.files.some(file => file.path === selected) ? selected : firstCaseFile(detail)) : '';
  const previewKey = id + '/' + chosen;
  const tree = useMemo(() => repositoryTree((detail?.files || []).filter(file => file.path.toLowerCase().includes(fileQuery.toLowerCase()))), [detail?.files, fileQuery]);
  function view(own: boolean) { const next = new URLSearchParams(); if (own) next.set('view', 'mine'); setParams(next); setOffset(0); }
  function openCase(value: string) { const next = new URLSearchParams(params); if (value) next.set('case', value); else next.delete('case'); setParams(next); }
  useEffect(() => { setOffset(0); }, [query, mine]);
  useEffect(() => {
    if (id) return;
    const abort = new AbortController(); setLoading(true); setError('');
    const timer = setTimeout(() => {
      void read<RsiCatalog>('/api/rsi-plaza/cases?' + new URLSearchParams({ query, view: mine ? 'mine' : 'all', offset: String(offset), limit: '24' }), abort.signal)
        .then(value => { if (!abort.signal.aborted) setCatalog(value); })
        .catch(() => { if (!abort.signal.aborted) setError(say('Cases could not be loaded.', '暂时无法加载案例。')); })
        .finally(() => { if (!abort.signal.aborted) setLoading(false); });
    }, query ? 180 : 0);
    return () => { abort.abort(); clearTimeout(timer); };
  }, [id, query, mine, offset, revision, lang]);
  useEffect(() => {
    setDetail(undefined); setSelected(''); setFileQuery(''); setDetailError(''); setDownloadError('');
    if (!id) return;
    const abort = new AbortController();
    void read<RsiCaseDetail>(caseEndpoint(id), abort.signal, id).then(value => { if (!abort.signal.aborted) setDetail(value); }).catch(() => { if (!abort.signal.aborted) setDetailError(say('This case is unavailable.', '暂时无法查看此案例。')); });
    return () => abort.abort();
  }, [id, revision, lang]);
  useEffect(() => {
    if (!chosen) return;
    const abort = new AbortController(); setPreview(undefined);
    void read<RsiPreview>(caseEndpoint(id, '/file?' + new URLSearchParams({ path: chosen })), abort.signal, id)
      .then(value => { if (!abort.signal.aborted) setPreview({ key: previewKey, value }); })
      .catch(() => { if (!abort.signal.aborted) setPreview({ key: previewKey, error: say('Preview unavailable. Download the repository to open this file.', '此文件暂不支持预览，可下载仓库后打开。') }); });
    return () => abort.abort();
  }, [id, chosen, revision, lang]);
  async function download() {
    if (!detail || downloading) return;
    setDownloading(true); setDownloadError('');
    try {
      const response = await fetch(caseEndpoint(id, '/download'), { credentials: 'same-origin', headers: getChatSessionAuthHeaders(caseSession(id)) });
      if (!response.ok) throw new Error();
      const url = URL.createObjectURL(await response.blob()), link = document.createElement('a');
      link.href = url; link.download = (detail.title.replace(/[^a-zA-Z0-9 _-]/g, '').trim().slice(0,80) || 'rsi-case') + '.zip'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch { setDownloadError(say('Download unavailable. Please try again.', '下载暂不可用，请重试。')); }
    finally { setDownloading(false); }
  }
  const retry = <button className="mp-button" onClick={() => setRevision(value => value + 1)}>{say('Retry', '重试')}</button>;
  if (id) return <div className="rsi-plaza rsi-case-page">
    <button className="rsi-back" onClick={() => openCase('')}>← {say('RSI Plaza', 'RSI 广场')}</button>
    {detailError ? <div role="alert" className="rsi-empty"><p>{detailError}</p>{retry}</div> : !detail ? <p role="status">{say('Loading repository…', '正在加载仓库…')}</p> : <>
      <header className="rsi-case-header"><div><h2>{detail.title}</h2><p>{detail.summary}</p><div className="rsi-meta"><span>{detail.visibility === 'private' ? say('Private', '仅自己') : say('Public', '公开')}</span>{detail.author_email && <a href={'mailto:' + encodeURIComponent(detail.author_email)}>{detail.author_email}</a>}<time>{caseDate(detail.created_at, lang)}</time></div></div><button className="mp-button primary" disabled={downloading} onClick={() => void download()}>{downloading ? say('Preparing…', '正在准备…') : say('Download ZIP', '下载 ZIP')}</button></header>
      {downloadError && <p role="alert" className="mp-error">{downloadError}</p>}
      <section className="rsi-repository" aria-label={say('Case repository', '案例仓库')}>
        <aside className="rsi-tree"><label><span>{say('Repository', '仓库文件')}</span><small>{detail.files.length} {say('files', '个文件')}</small></label><input type="search" aria-label={say('Find a file', '查找文件')} placeholder={say('Find a file…', '查找文件…')} value={fileQuery} onChange={event => setFileQuery(event.target.value)}/><FileTree node={tree} selected={chosen} choose={setSelected}/>{!tree.files.length && !tree.folders.length && <p>{say('No matching files', '没有匹配的文件')}</p>}</aside>
        <div className="rsi-preview"><header><span title={chosen}>{chosen || say('Repository', '仓库')}</span><div>{detail.readme && detail.files.some(file => file.path === detail.readme) && <button onClick={() => setSelected(detail.readme!)}>README</button>}{detail.report && detail.files.some(file => file.path === detail.report) && <button onClick={() => setSelected(detail.report!)}>{say('Report', '报告')}</button>}</div></header>
          {!chosen ? <p className="rsi-empty">{say('No preview files', '暂无预览文件')}</p> : preview?.key !== previewKey ? <p className="rsi-empty" role="status">{say('Loading file…', '正在加载文件…')}</p> : preview.error ? <p className="rsi-empty">{preview.error}</p> : preview.value?.binary ? <p className="rsi-empty">{say('Download ZIP to open this file.', '下载 ZIP 后打开此文件。')}</p> : <>{preview.value?.truncated && <p className="rsi-preview-note">{say('File preview · download ZIP for the complete file', '文件预览 · 下载 ZIP 查看完整文件')}</p>}{/\.md$/i.test(chosen) ? <div className="markdown-body" dangerouslySetInnerHTML={{ __html: renderCaseMarkdown(preview.value?.text || '') }}/> : <pre>{preview.value?.text}</pre>}</>}
        </div>
      </section>
    </>}
  </div>;
  return <div className="rsi-plaza">
    <header className="rsi-intro"><h2>{say('RSI Plaza', 'RSI 广场')}</h2><p>{say('Explore Scientist research: methods, code, evaluation and reports.', '浏览 Scientist 研究案例：方法、代码、评估与报告。')}</p></header>
    <div className="rsi-controls"><div className="mp-tabs"><button aria-pressed={!mine} className={!mine ? 'active' : ''} onClick={() => view(false)}>{say('All cases', '全部案例')}</button><button aria-pressed={mine} className={mine ? 'active' : ''} onClick={() => view(true)}>{say('My cases', '我的案例')}</button></div><input type="search" aria-label={say('Search research cases', '搜索研究案例')} placeholder={say('Search a protein, property or method…', '搜索蛋白、性质或方法…')} value={query} onChange={event => setQuery(event.target.value)}/></div>
    {error && <div role="alert" className="mp-error">{error} {retry}</div>}
    <div className="rsi-case-grid" aria-busy={loading}>{loading ? Array.from({length:4},(_,i)=><div key={i} className="rsi-case-skeleton" aria-hidden="true"/>) : catalog?.items.map(item => <button key={item.id} className="rsi-case-card" onClick={() => openCase(item.id)}><div className="rsi-card-top"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="M4 3h13a2 2 0 0 1 2 2v16H6a2 2 0 0 1-2-2Zm0 14h15M8 7h7M8 10h5"/></svg><h3>{item.title}</h3></div><p>{item.summary}</p><div className="rsi-meta"><span>{item.visibility === 'private' ? say('Private', '仅自己') : say('Public', '公开')}</span>{item.author_email && <span title={item.author_email}>{item.author_email}</span>}<time>{caseDate(item.created_at, lang)}</time></div><span className="rsi-card-action">{say('View repository', '查看仓库')} →</span></button>)}</div>
    {!loading && !error && !catalog?.items.length && <div className="rsi-empty"><h3>{query ? say('No matching cases', '没有匹配的案例') : mine ? say('Your research cases appear here', '你的研究案例将在这里展示') : say('No cases yet', '暂无案例')}</h3><p>{query ? say('Try another protein, property or method.', '试试其他蛋白、性质或方法。') : mine ? say('Start a topic in Scientist to develop a research project.', '在 Scientist 中开启课题，创建研究项目。') : say('Published research repositories will appear here.', '已发布的研究仓库将在这里展示。')}</p></div>}
    {!!catalog?.total && <footer className="rsi-pagination"><span>{catalog.total} {say('cases', '个案例')}</span><button className="mp-button" disabled={offset === 0 || loading} onClick={() => setOffset(value => Math.max(0, value - 24))}>← {say('Previous', '上一页')}</button><button className="mp-button" disabled={catalog.next_offset === null || catalog.next_offset === undefined || loading} onClick={() => setOffset(catalog.next_offset!)}>{say('Next', '下一页')} →</button></footer>}
  </div>;
}
