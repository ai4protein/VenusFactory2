"""Install the pinned optional reference library from Git or a source release."""
from pathlib import Path
import os
import re
import tempfile

from .common import capture, inside

PATH = 'src/agent/skills/vendor/scientific-agent-skills'


def install(root, run, metadata):
    revision = metadata.get('revision', '')
    repository = metadata.get('repository', '')
    if not re.fullmatch('[a-f0-9]{40}', revision) or not repository:
        raise ValueError('The optional skills library needs a pinned repository revision')
    target = inside(root, PATH)
    if (root / '.git').exists():
        run(['git', 'submodule', 'update', '--init', '--recursive', '--', PATH])
    elif (target / '.git').exists():
        # A source-archive installation must never overwrite a user's checkout.
        if capture(['git', '-C', target, 'status', '--porcelain']):
            raise ValueError('Optional skills have local modifications; preserve them before upgrading')
    else:
        if target.exists() and any(target.iterdir()):
            raise ValueError('Optional skills directory is not empty; preserve it before installing')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.skills-', dir=target.parent) as temporary:
            checkout = Path(temporary) / 'checkout'
            run(['git', 'clone', '--no-checkout', '--filter=blob:none', repository, checkout])
            run(['git', '-C', checkout, 'checkout', '--detach', revision])
            if target.exists():
                target.rmdir()  # Only the verified empty directory can be replaced.
            os.replace(checkout, target)
    actual = capture(['git', '-C', target, 'rev-parse', 'HEAD']).strip()
    if actual != revision:
        raise ValueError('Optional skills revision differs from the release manifest')
    if not (target / 'skills').is_dir():
        raise ValueError('Optional skills checkout has no skills directory')
    return {'status': 'ready', 'revision': actual, 'path': PATH}
