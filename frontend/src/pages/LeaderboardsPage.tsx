import { Link } from 'react-router-dom';
import { useLang } from '../lib/i18n';
import { useDocumentMeta } from '../lib/useDocumentMeta';
import { ModelGlyph } from '../components/ModelPlatformShared';

const boards = [
  { id: 'venusx', name: 'VenusX', category: 'residue', href: 'https://ai4protein.github.io/venusx/#leaderboard',
    scope: ['Functional understanding', '精细功能理解'],
    description: ['Compare models on residue identification, functional fragments and protein similarity.', '比较模型在功能残基识别、功能片段分类与蛋白相似性上的表现。'],
    tags: [['Residues', '残基'], ['Fragments', '片段'], ['Protein pairs', '蛋白配对']],
    query: '?category=residue' },
  { id: 'proteingym', name: 'ProteinGym', category: 'evolution', href: 'https://proteingym.org/benchmarks',
    scope: ['Mutation effects', '突变效应'],
    description: ['Compare protein fitness predictions against experimental mutation measurements.', '对照实验突变测量，比较蛋白适应度与突变效应预测模型。'],
    tags: [['Mutation scoring', '突变打分'], ['Fitness', '适应度']],
    query: '?category=evolution' },
];

export function LeaderboardsPage() {
  const { lang } = useLang(); const zh = lang === 'zh'; const index = zh ? 1 : 0;
  useDocumentMeta({ title: `${zh ? '排行榜' : 'Leaderboards'} — VenusFactory2`, description: 'VenusX and ProteinGym protein model benchmarks.' });
  return <div className="mp-leaderboards">
    <header className="mp-content-heading"><h2>{zh ? '比较模型表现' : 'Compare model performance'}</h2>
      <p>{zh ? '选择与你的研究任务相关的基准，查看公开评测结果。' : 'Choose a benchmark for your research task and explore published results.'}</p></header>
    <div className="mp-benchmark-grid">{boards.map(board => <article className="mp-benchmark-card" key={board.id}>
      <div className="mp-benchmark-title"><span className="mp-benchmark-icon"><ModelGlyph kind={board.category} size={22}/></span><h3>{board.name}</h3><span>{board.scope[index]}</span></div>
      <p>{board.description[index]}</p>
      <div className="mp-tags">{board.tags.map(tag => <span key={tag[0]}>{tag[index]}</span>)}</div>
      <div className="mp-benchmark-actions"><a className="mp-button primary" href={board.href} target="_blank" rel="noopener noreferrer">{zh ? '查看排行榜' : 'View leaderboard'} <span aria-hidden="true">↗</span></a>
        <Link className="mp-text-link" to={`/${lang}/models${board.query}`}>{zh ? '探索相关模型' : 'Explore related models'} →</Link></div>
    </article>)}</div>
  </div>;
}
