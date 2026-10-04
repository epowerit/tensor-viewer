# Contributing to TensorViewer

Thank you for helping. TensorViewer is two independent projects: `backend/` (Python, FastAPI, PyTorch) and `frontend/` (React, TypeScript, Vite). Most features touch both.

## Set up

Follow the [quick start](README.md#quick-start), or open the repository in GitHub Codespaces, which installs and starts both for you. You need Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and Node.js 22 or newer.

## Before you open a pull request

Run the same checks CI runs:

```sh
cd backend
uv run ruff check tensorviewer tests
uv run ruff format --check tensorviewer tests
uv run pytest -q
```

```sh
cd frontend
npm run format:check
npx tsc --noEmit -p .
npm test
npm run build
```

`uv run ruff format tensorviewer tests` and `npm run format` fix formatting.

## Conventions

- **Tests with the change.** A new rule, lens, or endpoint comes with a test that checks its numbers, not just that it runs. The backend tests use `TestClient` against a temporary data directory; the frontend tests are Vitest.
- **API types are generated.** After changing a backend model or endpoint, regenerate `frontend/openapi.json` from the app and run `npm run generate:api` in `frontend/`.
- **[How it works](docs/how-it-works.md) is the manual.** A user-visible change updates the section there that describes it, with a concrete example from a library project. The README stays a short overview: change it only when what TensorViewer is, or how to start it, changes.
- **Plain words in the interface.** Labels, notes, and tooltips say what something is and what to do about it, in full sentences, without jargon the user did not type.
- **Library projects are teaching material.** Files in `backend/tensorviewer/library/` keep their `# axes:` comments on the line they describe; the formatter leaves them alone.
- **Never commit data.** `backend/.data/` and other local data stay out of Git; `.gitignore` covers them.

## Security

The backend runs the code it is given. Keep it bound to `127.0.0.1`, and report a security problem privately to the maintainer rather than in a public issue.
