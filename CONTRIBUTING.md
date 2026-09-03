# Contributing to SPARQ

Thank you for your interest in improving SPARQ (`sparq-triage` on PyPI). Contributions of every size are welcome: bug reports, questions, documentation fixes, new test cases, and new features such as additional emitter platforms or estimator architectures.

## Reporting problems and asking for help

Open an issue at https://github.com/TaN-MM-Org/sparq-triage/issues. For bug reports, please include the package version (`python -c "import sparq; print(sparq.__version__)"`), a minimal script that reproduces the problem, and the output you expected. Usage questions are welcome in issues as well; there is no separate forum.

## Development setup

```
git clone https://github.com/TaN-MM-Org/sparq-triage
cd sparq-triage
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[test]
pytest tests -q
```

The suite runs in a few seconds and also runs in CI on every push and pull request. Without PyTorch the machine-learning tests skip automatically and the physics core is still fully tested.

## Design rules

* `sparq.physics`, `sparq.exact` and `sparq.pulsed` are the dependency-light core (NumPy/SciPy only) and must stay importable without PyTorch; a test enforces this.
* Every change to the physics must come with a test against the exact master-equation reference, a closed form, or brute-force numerics (the test suite shows the pattern for each).
* Estimator and environment changes must keep the shape contracts tested in `tests/test_torch_components.py`, or update those tests in the same commit with a stated reason.
* American spelling in prose.

## Governance and support

The package is maintained by Tanvir Mahmud Mahim (BRAC University), who reviews issues and pull requests. Releases are tagged on GitHub and published to PyPI by CI.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0 that covers the project.
