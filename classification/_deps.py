"""A readable error when the interpreter is missing the scientific stack.

This machine has several Python installations and only some carry numpy,
scikit-learn and pyriemann. Running an entry point with the wrong one produces a
bare ModuleNotFoundError partway through an import chain, which is confusing --
and for the server it looks like "connection refused" in the browser, because the
process died before it ever bound a port.

Entry points call :func:`require` first so the message says what is wrong and
which interpreter to use instead.
"""

import shutil
import subprocess
import sys

REQUIRED = ('numpy', 'scipy', 'sklearn', 'pyriemann', 'joblib')

# Interpreters worth suggesting, beyond whatever is on PATH.
CANDIDATE_PATHS = (
    '~/opt/anaconda3/bin/python3',
    '~/anaconda3/bin/python3',
    '~/miniconda3/bin/python3',
    '/opt/homebrew/bin/python3',
    '/usr/local/bin/python3',
)


def missing_packages():
    """List the required packages this interpreter cannot import.

    Returns
    -------
    list[str]
        Import names that failed, in the order of :data:`REQUIRED`.
    """

    import importlib.util

    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def find_working_interpreter():
    """Look for another interpreter on this machine that has everything.

    Returns
    -------
    str or None
        Path to a usable interpreter, or None if none was found.
    """

    import os

    seen = set()
    candidates = []

    for name in ('python', 'python3'):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    candidates.extend(os.path.expanduser(p) for p in CANDIDATE_PATHS)

    probe = 'import ' + ', '.join(REQUIRED)

    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real in seen or real == os.path.realpath(sys.executable):
            continue
        seen.add(real)

        if not os.access(candidate, os.X_OK):
            continue
        try:
            subprocess.run([candidate, '-c', probe], check=True, timeout=25,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            continue

        return candidate

    return None


def require(entry_point):
    """Exit with a useful message if this interpreter cannot run the code.

    Parameters
    ----------
    entry_point : str
        Module path of the caller, such as ``'classification.server'``, used to
        print a command the user can copy.

    Returns
    -------
    None
        Returns normally when everything is importable; otherwise exits.
    """

    missing = missing_packages()
    if not missing:
        return

    print(f'This Python is missing: {", ".join(missing)}', file=sys.stderr)
    print(f'  interpreter: {sys.executable}', file=sys.stderr)
    print(file=sys.stderr)

    working = find_working_interpreter()
    if working:
        print('Another interpreter on this machine has them. Use:', file=sys.stderr)
        print(f'    {working} -m {entry_point}', file=sys.stderr)
        print(file=sys.stderr)
        print('Or just run the launcher, which picks one for you:', file=sys.stderr)
        print('    ./run_ui.sh', file=sys.stderr)
    else:
        print('Install them first:', file=sys.stderr)
        print(f'    {sys.executable} -m pip install -r requirements.txt', file=sys.stderr)

    raise SystemExit(1)
