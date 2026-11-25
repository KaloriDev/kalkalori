# Contributing to KalKalori

Thank you for your interest in contributing to **KalKalori — Heat Exchanger Open Engine**.

KalKalori is an engineering-focused open-source project dedicated to
thermally and hydraulically correct modelling of heat exchangers.
Contributions are welcome from industry engineers, researchers, and students,
provided they align with the project’s technical and architectural principles.

---

## Project Philosophy

Before contributing, please familiarize yourself with the core ideas behind
KalKalori:

- Engineering correctness comes before feature breadth
- Physical assumptions must be explicit and documented
- Geometry defines flow topology — solvers do not infer it
- The core engine is intentionally free of UI, I/O, and serialization code
- Long-term maintainability is preferred over short-term convenience

---

## License and Legal Notice

KalKalori is released under the **GNU General Public License v3 (GPLv3 only)**.

By submitting a contribution, you confirm that:
- you have the right to submit the code or documentation,
- the contribution is not subject to conflicting intellectual property claims,
- the contribution will be licensed under GPLv3.

For substantial contributions, maintainers may request a brief statement of
authorship or origin.

---

## Scope of Contributions

### We Welcome

Contributions are particularly welcome in the following areas:

- verification and validation studies,
- corrections or improvements to heat transfer correlations,
- corrections or improvements to pressure drop models,
- documentation and example notebooks,
- code clarity, refactoring, and maintainability improvements.

Contributions aligned with the current roadmap have the highest chance of
acceptance.

---

### Out of Scope for the Core Repository

To keep KalKalori focused and reusable, the core repository intentionally
excludes:

- graphical user interfaces,
- file I/O or data serialization (JSON, Excel, databases),
- framework- or platform-specific integrations,
- proprietary datasets or manufacturer-specific correlations,
- undocumented heuristics or hard-coded empirical tuning.

Such functionality should live in **separate repositories or commercial
extensions**.

---

## Coding Guidelines

### General Style

- Python version: **3.10 or newer**
- Use explicit, descriptive variable and function names
- Prefer clarity over brevity
- Keep functions short and focused
- Avoid hidden defaults in geometry and topology definitions

### Units and Consistency

- **SI units only** are allowed in the core engine
- Units must be documented in code comments or docstrings
- Implicit unit conversions are not allowed

---

## Documentation and References

Built-in engineering correlations in KalKalori core must be independently
reproducible from legally and freely accessible source material. Models
requiring paywalled publications, licensed datasets, proprietary software,
or manufacturer-confidential information are integrated through external
providers. This is a project source policy, independent of whether the
underlying mathematical method could otherwise be implemented legally.

- Every non-trivial formula must include a **literature reference** in comments
- Assumptions and applicability limits must be stated explicitly
- Avoid undocumented “magic numbers”
- Do not commit commercial standards, books, journal PDFs, or other
  third-party full-text reference works unless their redistribution license
  explicitly permits inclusion in the repository. Store only bibliographic
  references (see `docs/references.md`) and obtain source material
  separately through legitimate access.

Example:
# Darcy–Weisbach equation
# Ref: White, F. M., "Fluid Mechanics"

## Versioning and Compatibility

KalKalori follows **Semantic Versioning (SemVer)**.

- Changes affecting numerical results or public APIs must be documented in
  `CHANGELOG.md`
- Breaking changes should be avoided before version `1.0.0`
- Any unavoidable breaking change must be clearly explained
- The authoritative import/build version is `core.__version__` in
  `core/__init__.py`; `pyproject.toml` derives the distribution version from it.
  Do not add independent version declarations or permanently patch generated
  package metadata. Git tags and the changelog record release history;
  completed development does not itself establish a publication date.

### Release metadata checklist

1. Verify the latest merged default branch and create a dedicated development
   branch. Preserve unrelated work; do not prepare release corrections directly
   on the default branch or a detached HEAD.
2. Reconcile the implemented scope with Git history and validation evidence.
   Update the canonical version, separate CHANGELOG entries, README current
   version references and roadmap. Preserve historical receipts and documented
   model limitations; record dates only when supported.
3. In the intended runtime environment, regenerate distribution metadata with
   the existing setuptools editable-install workflow (build requirements must
   already be available):

   ```powershell
   python -c "from setuptools import setup; setup(script_args=['egg_info'])"
   python -m pip install --no-index --no-deps --no-build-isolation -e .
   python tools/check_release_metadata.py
   ```

   The egg-info command regenerates source-tree metadata that KON can find on
   its repository search path; the editable install refreshes the environment's
   distribution metadata. The check compares repository/runtime/distribution versions, the dynamic
   version source, CHANGELOG, current README/roadmap references, imported module
   location and the development branch. Use `--expected-branch <name>` to also
   require the intended branch. It fails on discrepancies rather than rewriting
   them. Run it in a fresh process after committing as well.
4. Verify relevant import/build and regression checks. For private KON work,
   refresh `run_metadata` and read back the Excel worksheet in its normal
   environment, including Git SHA/branch/dirty status. Preserve discrepancy
   detection and distinguish ignored notebook edits from Git dirty state.
5. Commit only the reviewed scope on the dedicated branch. Push, PR creation,
   merge, tagging and publication require their own authorization.

---

## Development Workflow

1. Fork the repository
2. Create a dedicated branch for your change
3. Add or update tests and/or notebooks where appropriate
4. Ensure the code is readable and well-documented
5. Submit a pull request with a clear technical description

---

## Review Process

Pull requests are evaluated based on:

- correctness of physics and numerical methods
- consistency with project architecture
- clarity and maintainability of the code
- quality of documentation and references

Not all technically correct contributions will be accepted if they conflict
with the long-term design goals of the project.

---

## Academic and Industrial Contributions

KalKalori explicitly encourages:

- student and academic projects
- independent validation studies
- industrial feedback and corrections

Contributors working on behalf of a company or institution should ensure that
intellectual property and licensing requirements are satisfied before
submission.

---

## Final Note

KalKalori aims to grow deliberately and responsibly.

Well-documented, carefully reasoned contributions are valued more than rapid
feature expansion.

