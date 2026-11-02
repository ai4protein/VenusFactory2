import { Link } from 'react-router-dom';
import { useLang } from '../../lib/i18n';
import { MANUAL_SECTIONS, manualChatHref } from '../../lib/manualCatalog';

export function ManualIndexPage() {
  const { lang } = useLang();
  const zh = lang === 'zh';
  return <section className="manual-v2-panel manual-guide-index">
    <aside className="manual-v2-nav"><ul>
      <li><a className="manual-v2-nav-link" href="#choose-mode">{zh ? '选择工作方式' : 'Choose a mode'}</a></li>
      <li><a className="manual-v2-nav-link" href="#guides">{zh ? '按任务查阅' : 'Find a guide'}</a></li>
    </ul></aside>
    <div className="manual-v2-content-wrap"><article className="manual-v2-content manual-v2-markdown">
      <section id="choose-mode" className="manual-guide-intro">
        <h1>{zh ? '从一个蛋白质问题开始' : 'Start with a protein question'}</h1>
        <p>{zh ? '查看教程，或直接向 Agent 询问平台用法。' : 'Follow a tutorial or ask Agent how to use the platform.'}</p>
        <div className="manual-guide-actions">
          <Link className="manual-guide-start" to={`/${lang}/manual/start`}>{zh ? '快速上手' : 'Getting started'} →</Link>
          <Link className="manual-guide-ask" to={manualChatHref('start',lang)}>{zh ? '询问 Agent' : 'Ask Agent'} ↗</Link>
        </div>
        <div className="manual-guide-modes">{MANUAL_SECTIONS.filter(s => ['agent','scientist','researcher'].includes(s.key)).map(s =>
          <Link className="manual-guide-mode" key={s.key} to={`/${lang}/manual/${s.key}`}><h2>{s.title[lang]}</h2><p>{s.summary[lang]}</p><span>{zh ? '查看教程' : 'Read guide'} →</span></Link>
        )}</div>
      </section>
      <section id="guides"><h2>{zh ? '按任务查阅' : 'Find a guide'}</h2>
        <div className="manual-guide-list">{MANUAL_SECTIONS.filter(s => !['agent','scientist','researcher'].includes(s.key)).map(s =>
          <Link key={s.key} to={`/${lang}/manual/${s.key}`}><strong>{s.title[lang]}</strong><p>{s.summary[lang]}</p><span aria-hidden="true">↗</span></Link>
        )}</div>
      </section>
    </article></div>
  </section>;
}
