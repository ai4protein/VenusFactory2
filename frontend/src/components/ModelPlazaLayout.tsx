import { Link, Navigate, Outlet, useLocation } from 'react-router-dom';
import { useLayoutEffect, useRef } from 'react';
import { useLang } from '../lib/i18n';
import { modelPlazaSection } from '../lib/modelPlazaNavigation';
import '../model-platform.css';
import '../model-plaza-workspace.css';

const sections = [
  ['', 'Explore', '工具与模型'], ['rsi', 'RSI Plaza', 'RSI 广场'], ['training', 'Train', '训练'],
  ['evaluation', 'Evaluate', '评估'], ['predict', 'Predict', '预测'],
  ['leaderboards', 'Leaderboards', '排行榜'],
];

export function ModelPlazaLayout() {
  const { lang } = useLang(); const zh = lang === 'zh';
  const { pathname } = useLocation();
  const active = modelPlazaSection(pathname);
  const tabular = pathname.endsWith('/new');
  const shell = useRef<HTMLElement>(null);
  const nav = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    shell.current?.scrollIntoView({ block: 'start' });
    const selected = nav.current?.querySelector<HTMLElement>('[aria-current="page"]');
    if (selected && nav.current) {
      const container = nav.current.getBoundingClientRect();
      const item = selected.getBoundingClientRect();
      if (item.right > container.right) nav.current.scrollLeft += item.right - container.right;
      else if (item.left < container.left) nav.current.scrollLeft -= container.left - item.left;
    }
  }, [pathname]);
  return <section className="mp-shell" ref={shell}>
    <header className="mp-shell-header"><h1>{zh ? '模型广场' : 'Model Plaza'}</h1>
      <p>{zh ? '探索蛋白工具，用自己的数据训练、评估和使用模型。' : 'Explore protein tools. Train, evaluate and use models with your data.'}</p>
    </header>
    <nav ref={nav} className="mp-section-nav" aria-label={zh ? '模型广场导航' : 'Model plaza navigation'}>
      {sections.map(([path, en, cn]) => <Link key={path} to={`/${lang}/models${path ? '/' + path : ''}`}
        aria-current={active === path ? 'page' : undefined}>{zh ? cn : en}</Link>)}
    </nav>
    {active === 'training' && <nav className="mp-training-nav" aria-label={zh ? '训练方式' : 'Training approach'}>
      <Link to={`/${lang}/models/training`} aria-current={!tabular ? 'page' : undefined}>{zh ? '蛋白模型微调' : 'Protein model fine-tuning'}</Link>
      <Link to={`/${lang}/models/new`} aria-current={tabular ? 'page' : undefined}>{zh ? '实验数据建模' : 'Experimental data model'}</Link>
    </nav>}
    <Outlet />
  </section>;
}

export function LegacyModelRedirect({ to }: { to: string }) {
  const { lang } = useLang(); const { search, hash } = useLocation();
  return <Navigate to={`/${lang}${to}${search}${hash}`} replace />;
}
