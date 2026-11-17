"""Build optional CPU kernels with the current environment's Python/compiler.

Run from the project root: .venv/bin/python -m work.build_native
No runtime dependency is installed, and imports never trigger a compiler.
"""
from pathlib import Path
from setuptools import Extension, setup


def main():
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root:
        raise SystemExit('Run this command from the repository root')
    setup(name='wtf-native-kernels', version='0.0.0', packages=['ecosystem'],
          ext_modules=[Extension('ecosystem._kernels', ['ecosystem/_kernels.c'],
                                 extra_compile_args=['-O3', '-fno-fast-math',
                                                     '-ffp-contract=off', '-fno-builtin-pow'])],
          script_args=['build_ext', '--inplace', '--build-temp', '.cache/native-build/temp',
                       '--build-lib', '.cache/native-build/lib'])


if __name__ == '__main__':
    main()
