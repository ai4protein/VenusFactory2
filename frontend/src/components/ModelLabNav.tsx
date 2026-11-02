import { LangLink } from './LangLink';
import { useLang } from '../lib/i18n';
export function ModelLabNav({active}:{active:'training'|'evaluation'|'predict'}) {
  const {lang}=useLang();const zh=lang==='zh';
  return <nav className="model-lab-nav" aria-label={zh?'模型开发':'Model development'}>
    <div>{[['training','Train','训练'],['evaluation','Evaluate','评估'],['predict','Predict','预测']].map(([id,en,cn])=><LangLink key={id} to={`/models/${id}`} className={active===id?'active':''} aria-current={active===id?'page':undefined}>{zh?cn:en}</LangLink>)}</div>
    <LangLink to="/models?view=mine">{zh?'我的模型':'My models'} ↗</LangLink>
  </nav>;
}
