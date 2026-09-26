# Contributing

Thanks for helping build woyo.

- **Issues**: bug reports and task-capability requests are both welcome — describe the *goal*, not just the symptom.
- **PRs**: keep them scoped; run `pytest` and `ruff check src tests` first. New tools must include tests and declare an honest `Permission`.
- **Security**: do not open public issues for vulnerabilities — use the GitHub security advisory flow (see docs/SECURITY.md).
- **Design changes**: anything touching the agent loop, security model, or public interfaces should come with a short ADR in docs/DECISIONS.md.

Style: Python 3.11+, full type hints, asyncio-first, stdlib before dependencies, dependencies before frameworks.
