import type { ModelCard } from './modelPlatform';
export function unavailableLabel(model: ModelCard, zh: boolean) {
  if (model.runtime_mode === 'online' || model.disabled_reason === 'local_only') return zh ? '线上无法提供' : 'Unavailable online';
  if (model.disabled_reason === 'not_configured') return zh ? '暂未配置' : 'Not configured';
  return zh ? '暂未开启' : 'Currently disabled';
}
