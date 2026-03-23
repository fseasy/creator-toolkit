# Agent Guidelines for creator-toolkit

This repository contains a monorepo with:
- **Backend**: Python projects (TTS, audio-sep) using FastAPI
- **Frontend**: React + TypeScript + Vite application

## Project Structure

```
creator-toolkit/
├── backend/
│   ├── audio-sep/          # Audio separation service
│   └── tts/
│       ├── chatterbox-turbo/
│       └── qwen3-tts/      # Qwen3 TTS server
├── frontend/               # React TypeScript app
└── README.md
```

---

## Backend (Python)

### Environment Setup

```bash
# Navigate to project
cd backend/tts/qwen3-tts  # or backend/audio-sep

# Install dependencies (uses uv)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Build/Lint/Test Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies |
| `uv add <package>` | Add a new dependency |
| `uv run python main.py` | Run main.py |
| `uv run ruff check .` | Lint with ruff |
| `uv run ruff check --fix .` | Auto-fix lint issues |
| `uv run ruff format .` | Format code |
| `uv run ruff check <file>` | Lint single file |
| `uv run ruff check --fix <file>` | Fix single file |

### Code Style Guidelines

**Python Version**: 3.14

**Linter**: Ruff (configured in pyproject.toml)

**Rules Enabled**:
- E, F: pycodestyle, pyflakes
- B: flake8-bugbear
- UP: pyupgrade
- I: isort
- RET: return statements

**Formatting**:
- Line length: 120
- Indent: 2 spaces (for this project)
- Use `uv` for dependency management

**Imports** (isort):
- `force-single-line`: false
- `combine-as-imports`: true

**Naming Conventions**:
- Modules: snake_case (`fs_qwen3_tts_server`)
- Functions: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE
- Use pathlib.Path for paths (not os.path or string paths)

**Code Guidelines**:
- Use type hints for all function parameters and return values
- Comments in English, keep minimal
- Prefer f-strings for string formatting
- Use dataclasses or Pydantic for structured data
- FastAPI routes: use `Depends()` (B008 warnings suppressed for routes)

**Error Handling**:
- Use custom exceptions with clear messages
- Return appropriate HTTP status codes in FastAPI endpoints
- Log errors with context using standard logging

---

## Frontend (React + TypeScript)

### Environment Setup

```bash
cd frontend
pnpm install
```

**Node**: >=20 <21 (see .nvmrc)

### Build/Lint/Test Commands

| Command | Description |
|---------|-------------|
| `pnpm dev` | Start dev server with HMR |
| `pnpm build` | Type-check and build for production |
| `pnpm lint` | Run ESLint |
| `pnpm lint --fix` | Auto-fix lint issues |
| `pnpm preview` | Preview production build |
| `pnpm exec eslint <file>` | Lint single file |
| `pnpm exec tsc --noEmit` | Type-check without build |

### Code Style Guidelines

**TypeScript**: Strict mode enabled (tsconfig.app.json)

**Compiler Options**:
- Target: ES2022
- Module: ESNext
- JSX: react-jsx
- Strict: true
- `noUnusedLocals`: true
- `noUnusedParameters`: true

**Linter**: ESLint flat config with:
- @eslint/js (recommended)
- typescript-eslint (recommended)
- eslint-plugin-react-hooks
- eslint-plugin-react-refresh

**Imports**:
- Use absolute imports (no `./` prefixes for project imports)
- Group imports: React > third-party > local
- Use `verbatimModuleSyntax` (no type-only imports shortcuts)

**Naming Conventions**:
- Components: PascalCase (`App.tsx`, `MyComponent.tsx`)
- Hooks: camelCase with `use` prefix (`useAuth.ts`)
- Utilities: camelCase
- CSS files: Match component name (`App.css`)

**React Patterns**:
- Use functional components with hooks
- Prefer composition over prop drilling
- Use React 19 features when appropriate
- Export components as named exports or default

**TypeScript Patterns**:
- Use `interface` for object shapes, `type` for unions/complex types
- Avoid `any` - use `unknown` when type is truly unknown
- Use explicit return types for complex functions
- Props interfaces should be named `ComponentNameProps`

**File Organization**:
```
src/
├── components/      # Reusable UI components
├── hooks/           # Custom React hooks
├── utils/           # Utility functions
├── types/           # TypeScript type definitions
├── App.tsx          # Root component
└── main.tsx         # Entry point
```

**Error Handling**:
- Use Error Boundaries for component errors
- Handle async operations with proper loading/error states
- Use Zod or similar for runtime validation

---

## General Guidelines

1. **No .env files in commits** - Use environment variables, never commit secrets
2. **Test your changes** - Run lint and type-check before submitting
3. **Minimal comments** - Code should be self-documenting
4. **Consistent patterns** - Follow existing code conventions in each project
5. **Path handling** - Always use pathlib (Python) or proper path utilities (TS)

## Running Specific Tests

There are no dedicated test files currently. For testing:
- Python: Run `uv run python <file.py>` to test functionality
- Frontend: Use `pnpm dev` to test in browser with hot reload
