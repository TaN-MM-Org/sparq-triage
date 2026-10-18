"""The worked tutorials in examples/ run and print what their docstrings
(and the README) show.

Each script is run in a fresh Python process. Its output must have the
same lines and words as the output quoted at the top of the script, and
every printed number must agree with the quoted one to 1 % (or 0.002),
so that the check also passes with other NumPy and SciPy versions, whose
optimizers can stop at slightly different points (the quoted output is
from Python 3.13, NumPy 2.5.3 and SciPy 1.18.1, and is exact there)."""
import ast
import os
import pathlib
import re
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
NUM = re.compile(r"(?<![A-Za-z_])(?:[-+]?(?:\d+\.\d*|\.\d+|\d+)"
                 r"(?:[eE][-+]?\d+)?|nan|inf)(?![A-Za-z_])")


def _quoted_output(path):
    doc = ast.get_docstring(ast.parse(path.read_text()))
    head, sep, out = doc.partition("SciPy 1.18.1):\n\n")
    assert sep, "the docstring must quote the printed output"
    return out.strip("\n")


def _check(path):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    run = subprocess.run([sys.executable, str(path)], capture_output=True,
                         text=True, timeout=600, env=env,
                         cwd=str(path.parents[1]))
    assert run.returncode == 0, run.stderr
    got = run.stdout.strip("\n").splitlines()
    want = _quoted_output(path).splitlines()
    assert len(got) == len(want), run.stdout
    for g, w in zip(got, want):
        assert NUM.sub("#", g) == NUM.sub("#", w), (g, w)
        for a, b in zip(NUM.findall(g), NUM.findall(w)):
            a, b = float(a), float(b)
            if b != b:                                  # nan
                assert a != a, (g, w)
            else:
                assert abs(a - b) <= max(0.01 * abs(b), 2e-3), (g, w)


@pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ not present")
@pytest.mark.parametrize("name", ["tutorial_saturation.py",
                                  "tutorial_timetags.py",
                                  "tutorial_lifetime.py"])
def test_tutorial_runs_and_prints_the_quoted_numbers(name):
    _check(EXAMPLES / name)
