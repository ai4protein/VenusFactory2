import { registerHooks } from 'node:module';
import { existsSync } from 'node:fs';
import { extname } from 'node:path';
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL?.startsWith('file:') && /^\.{1,2}\//.test(specifier) && !extname(specifier)) {
      const candidate = new URL(specifier + '.ts', context.parentURL);
      if (existsSync(candidate)) return nextResolve(candidate.href, context);
    }
    return nextResolve(specifier, context);
  },
});
