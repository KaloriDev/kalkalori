"""Read-only release identity check; run in the intended installed environment.

Usage: python tools/check_release_metadata.py [--expected-branch NAME]
The canonical version comes from core/__init__.py, never from this tool.
"""
from __future__ import annotations

import argparse
import ast
from email.parser import Parser
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def check(expected_branch: str | None = None) -> dict:
    errors = []
    identity = {}

    def require(condition, message):
        if not condition:
            errors.append(message)

    def git(*arguments):
        return subprocess.check_output(
            ['git', *arguments], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=10,
        ).strip()

    source = ROOT / 'core' / '__init__.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    declarations = [node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == '__version__'
                            for target in node.targets)]
    require(len(declarations) == 1, 'Expected one authoritative core.__version__ declaration.')
    version = ast.literal_eval(declarations[0].value) if len(declarations) == 1 else None
    identity['repository_version'] = version
    require(isinstance(version, str) and bool(version), 'Canonical version must be a nonempty string.')

    # Validate this repository's existing simple dynamic configuration without
    # adding a TOML dependency or requiring Python newer than the core supports.
    pyproject = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    project = re.search(r'(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)', pyproject)
    dynamic = re.search(r'(?m)^dynamic\s*=\s*(\[[^\]]*\])', project[1]) if project else None
    require(bool(dynamic) and 'version' in ast.literal_eval(dynamic[1]),
            'pyproject.toml must retain dynamic project version metadata.')
    require(not project or not re.search(r'(?m)^version\s*=', project[1]),
            'Do not add an independent project.version declaration.')
    configuration = re.search(r'(?ms)^\[tool\.setuptools\.dynamic\]\s*\n(.*?)(?=^\[|\Z)', pyproject)
    require(bool(configuration) and bool(re.search(
        r'''(?m)^version\s*=\s*\{\s*attr\s*=\s*["']core\.__version__["']\s*\}\s*$''',
        configuration[1],
    )), 'Setuptools version must derive from core.__version__.')

    try:
        core = importlib.import_module('core')
        identity['runtime_version'] = getattr(core, '__version__', None)
        identity['imported_core'] = str(Path(core.__file__).resolve())
        require(identity['runtime_version'] == version, 'Runtime and repository versions differ.')
        require(Path(core.__file__).resolve() == source.resolve(), 'Runtime imports a different core repository.')
    except ImportError as error:
        errors.append(f'Cannot import installed core: {error}')

    try:
        identity['distribution_version'] = importlib.metadata.version('kalkalori')
        require(identity['distribution_version'] == version, 'Installed distribution and repository versions differ.')
        distributions = {str(Path(distribution._path).resolve()): distribution.version
                         for distribution in importlib.metadata.distributions(name='kalkalori')}
        identity['distribution_metadata'] = distributions
        require(bool(distributions) and all(value == version for value in distributions.values()),
                'A discoverable kalkalori distribution has stale version metadata.')
    except importlib.metadata.PackageNotFoundError:
        errors.append('KalKalori distribution metadata is missing; regenerate it with the documented install workflow.')

    # A notebook adds the repository to sys.path. Its generated egg-info can
    # shadow an otherwise current site-packages distribution in that context.
    local_metadata = ROOT / 'kalkalori.egg-info' / 'PKG-INFO'
    if local_metadata.is_file():
        local_version = Parser().parsestr(local_metadata.read_text(encoding='utf-8')).get('Version')
        identity['source_tree_distribution_version'] = local_version
        require(local_version == version, 'Source-tree egg-info is stale and can shadow installed metadata in notebooks.')

    changelog = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    entries = re.findall(r'(?m)^## \[([^\]]+)\]', changelog)
    require(bool(entries) and entries[0] == version, 'The latest CHANGELOG entry must match the canonical version.')
    require(entries.count(version) == 1, 'CHANGELOG must contain exactly one current-version entry.')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    headings = re.findall(r'(?m)^## Current Capabilities \(v([^\)]+)\)', readme)
    statements = [value.rstrip('.,') for value in
                  re.findall(r'The current package version is v(\S+)', readme)]
    require(bool(headings) and bool(statements)
            and all(value == version for value in headings + statements),
            'README current-version references must match the canonical version.')
    roadmap = (ROOT / 'docs' / 'roadmap.md').read_text(encoding='utf-8')
    current = re.findall(r'\*\*Current version:\*\*\s*`v([^`]+)`', roadmap)
    require(len(current) == 1 and current[0] == version, 'Roadmap current version must match the canonical version.')

    try:
        branch = git('branch', '--show-current')
        identity.update(git_branch=branch, git_head=git('rev-parse', 'HEAD'),
                        git_dirty=bool(git('status', '--porcelain=v1')))
        try:
            default_branch = git('symbolic-ref', '--short', 'refs/remotes/origin/HEAD').removeprefix('origin/')
        except subprocess.CalledProcessError:
            default_branch = 'main'
        identity['default_branch'] = default_branch
        require(bool(branch) and branch not in {default_branch, 'main', 'master'},
                'Release preparation requires a dedicated development branch, not the default branch or detached HEAD.')
        if expected_branch is not None:
            require(branch == expected_branch, f'Expected development branch {expected_branch!r}, got {branch!r}.')
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f'Cannot verify Git development branch: {error}')

    return {'status': 'FAIL' if errors else 'PASS', **identity, 'errors': errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-branch')
    args = parser.parse_args()
    result = check(args.expected_branch)
    print(json.dumps(result, indent=2))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
