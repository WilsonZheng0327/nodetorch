import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // `any` is a deliberate choice at NodeTorch's unstructured boundaries, not
      // an oversight: `ExecutionResult.metadata` (each node emits whatever its
      // visualization needs) and node property `value`s are untyped by design —
      // see "Key Design Decisions" in CLAUDE.md. Forcing concrete types there
      // would fight the architecture for no real safety gain. So keep
      // no-explicit-any visible as a warning (to flag accidental new `any`s in
      // ordinary code) rather than a build-failing error.
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  },
])
