"""Prepare a validated release, then publish it in one database transaction."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-stats', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without writing or connecting')
    parser.add_argument('--release-dir', help='Validate and publish an existing release')
    parser.add_argument('--include-experimental', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    visualizer = Path(os.environ.get('PEDX_VISUALIZER_ROOT', root.parent / 'PedX-Visualizer'))
    release_root = Path(os.environ.get('PEDX_RELEASES_ROOT', Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'PedX' / 'releases'))
    release = Path(args.release_dir).resolve() if args.release_dir else release_root / uuid.uuid4().hex
    commands = []
    if not args.release_dir:
        command = [sys.executable, '-B', '-m', 'scripts.build_release', '--output', str(release)]
        if args.skip_stats:
            command.append('--skip-stats')
        if args.include_experimental:
            command.append('--include-experimental')
        commands.append((command, root))
    commands.append((['node', 'scripts/publish-release.js', '--csv-dir', str(release)], visualizer))
    for command, cwd in commands:
        print(f'[publish] {subprocess.list2cmdline(command)} (cwd={cwd})', flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=cwd, check=True)


if __name__ == '__main__':
    main()
