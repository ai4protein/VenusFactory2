type Scope = 'platform' | 'custom';
const pending: Partial<Record<Scope, Promise<void>>> = {};

// Parallel page loads share one bootstrap request, so they cannot replace
// each other's freshly issued browser identity.
export function ensureJobSession(scope: Scope, initialize?: () => Promise<void>): Promise<void> {
  if (!pending[scope]) {
    const path = scope === 'platform' ? '/api/model-platform/session' : '/api/custom-model/session';
    pending[scope] = (initialize ? initialize() : fetch(path, { credentials: 'same-origin' }).then(response => {
      if (!response.ok) throw new Error('Workspace unavailable');
    })).catch(error => { delete pending[scope]; throw error; });
  }
  return pending[scope]!;
}
