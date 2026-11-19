// ESLint flat config for the operator UI.
//
// Rule sets: @eslint/js recommended, typescript-eslint recommended (syntactic),
// and eslint-plugin-react-hooks recommended (rules-of-hooks, exhaustive-deps
// and the React Compiler purity/refs/effect rules). One type-aware rule is
// added on top: no-floating-promises, because a dropped promise on a safety
// path (E-stop, disarm, lease release) must be an explicit `void` decision.
import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores(["dist/", "coverage/", "node_modules/"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
    ],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_", caughtErrors: "none" },
      ],
    },
  },
  {
    files: ["*.config.{js,ts}"],
    languageOptions: { globals: globals.node },
  },
  {
    files: ["e2e/*.mjs"],
    extends: [js.configs.recommended],
    languageOptions: { ecmaVersion: 2022, globals: globals.node },
  },
  {
    files: ["eslint.config.js"],
    extends: [tseslint.configs.disableTypeChecked],
  },
]);
