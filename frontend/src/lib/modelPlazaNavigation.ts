/** Keep model development inside the plaza while retaining old shared links. */
export function modelPlazaSection(pathname: string): string {
  const page = pathname.replace(/\/$/, '').split('/models')[1]?.replace(/^\//, '') || '';
  return page === 'new' ? 'training' : page;
}

export function modelWorkspaceRoute(model: { category?: string; name: string; route?: string }): string {
  if (model.category === 'foundation') {
    const name = model.name.replace(/\s*·\s*Backbone$/, '');
    return '/models/training?' + new URLSearchParams({ plm: name });
  }
  return (model.route || '/models/predict').replace(/^\/custom-model\//, '/models/');
}

export function initialProteinModel(requested: string | null, allowed: Record<string, string>): string {
  return requested && Object.prototype.hasOwnProperty.call(allowed, requested) ? requested : Object.keys(allowed)[0] || '';
}
