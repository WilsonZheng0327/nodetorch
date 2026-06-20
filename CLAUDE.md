# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NodeTorch — node-based visual tool for building, inspecting, and understanding ML models. Educational, open-source. Target users are students learning ML.

## Commands

```bash
# Frontend
npm install              # install dependencies
npm run dev              # dev server at http://localhost:5173
npm run build            # type-check + production build (tsc -b && vite build)
npx tsc --noEmit         # type-check only
npm run lint             # ESLint (flat config, TS/TSX only)

# Backend (from repo root, uses .venv)
.venv/bin/python backend/main.py   # FastAPI server at http://localhost:8000

# Tests — frontend (vitest), backend (pytest)
npm test                                          # all frontend tests
npx vitest run tests/frontend/core/graph.test.ts  # single frontend test file
.venv/bin/pytest                                  # all backend tests
.venv/bin/pytest tests/backend/test_export.py     # single backend test file
```

Test locations: `tests/frontend/` (vitest, configured in `vite.config.ts`) and `tests/backend/` (pytest, configured in `pytest.ini` with `pythonpath = backend`).

Backend must be restarted manually after Python changes — no hot reload.

## Architecture

6-layer design. Each layer only depends on layers below it. **Layers 1-4 know nothing about ML.**

```
Layer 6: UI          → src/ui/           (React + React Flow)
Layer 5: ML Domain   → src/domain/       (register() calls — all ML knowledge lives here)
Layer 4: Node Registry → src/core/nodedef.ts
Layer 3: Execution Engine → src/core/engine.ts
Layer 2: Type System → src/core/datatypes.ts
Layer 1: Graph Core  → src/core/graph.ts
```

Backend mirrors the engine: `backend/engine/graph_builder/` does the same topological walk but with real PyTorch tensors.

### Key files

- `src/core/graph.ts` — NodeInstance, Edge, Graph, topological sort, dirty tracking
- `src/core/nodedef.ts` — NodeDefinition interface, NodeRegistry, PortDefinition, PropertyDefinition
- `src/core/engine.ts` — ExecutionEngine, ExecutionModeDefinition, Executor interface
- `src/core/validation.ts` — pre-flight checks for forward/training (standalone, no side effects)
- `src/domain/index.ts` — bootstraps all registrations via `initDomain()`
- `src/ui/useGraph.ts` — bridge between engine and React Flow (the main state hook)
- `src/ui/EngineNode.tsx` — generic node renderer (reads NodeDefinition + lastResult.metadata)
- `src/ui/step-through/` — forward + backward step-through UI (StepThroughPanel, StageDetail, ExtraPanels)
- `src/ui/dashboard/TrainingDashboard.tsx` — training dashboard with charts, metrics, system info
Backend is organized into packages by concern (all importable with `pythonpath = backend`):

- `backend/engine/` — execution engine (ML-agnostic graph→PyTorch core)
  - `engine/graph_builder/` — graph→PyTorch execution, split into submodules: `constants` (node-type sets), `_state` (device + in-memory model store, shared mutable state), `stats` (tensor/param statistics), `build` (topo sort, input routing, module/subgraph construction), `forward` (single forward pass), `detail` (per-node inspector viz), `inference` (infer + test-set eval + tracked-sample helpers). The package `__init__.py` re-exports the full public API, so `from engine.graph_builder import X` is unchanged.
  - `engine/node_builders.py` — per-node-type `nn.Module` builder functions
  - `engine/forward_utils.py` — shared single-node / forward-pass execution helpers
- `backend/dataprep/` — `data_loaders.py` (per-dataset loaders), `bpe.py` (BPE tokenizer, cached per dataset+vocab), `tokenizer_preview.py`. Named `dataprep` (not `datasets`) to avoid shadowing the HuggingFace `datasets` library.
- `backend/visualize/` — feature-level visualizations: `node_viz.py` (aggregates per-layer viz into `FORWARD_VIZ`/`BACKWARD_VIZ` registries), `step_through.py` (forward step-through), `backprop_sim.py` (backward step-through), `denoise_viz.py` (diffusion denoising), `latent_viz.py` (VAE latent grid), `activation_max.py`, `loss_landscape.py`
  - `visualize/layers/` — per-layer viz functions, one file per layer family (conv, linear, activation, norm, pool, etc.)
- `backend/generate/` — `text_generate.py` (autoregressive char/BPE generation), `gan_generate.py` (GAN image generation on demand)
- `backend/export/` — generates standalone runnable PyTorch training scripts from graphs; entry point `from export import export_to_python`. Split into `templates`, `helpers`, `layers`, `model`, `training_loops`, `datasets`, `exporter`.
- `backend/persistence/` — `runs_store.py` (training-run persistence)
- `backend/training/` — training loop plugin system (standard, GAN, diffusion, autoregressive)
- `src/ui/tutorial/` — guided tutorial system (goal-based tasks with auto-detection)
- `model-presets/` — shipped preset graph JSON files, served via `GET /presets` endpoint

### Styling

Plain CSS with CSS custom properties for theming. No Tailwind, no CSS modules. Theme variables defined in `src/index.css` (Catppuccin-inspired palette). Dark theme is default; light theme uses `[data-theme="light"]` selector. Component styles are colocated CSS files (e.g., `EngineNode.css`).

### Frontend ↔ Backend communication

- **Shape mode**: no backend needed — pure TypeScript math.
- **Training**: WebSocket at `ws://localhost:8000/ws`. Frontend sends serialized graph JSON. Backend streams messages: `{ type: 'epoch', epoch, loss, accuracy?, ...metrics }`, `{ type: 'loss', loss }` (per-step), `{ type: 'done', result }`. Frontend can send `{ type: 'cancel' }`.
- **Other modes** (infer, test, step-through, export): REST endpoints on the FastAPI server.
- **Graph serialization**: `SerializedGraph` format (version `'1.0'`) with nested `subgraph` support for composite nodes. Serialize/deserialize functions in `useGraph.ts`.

### Execution modes

- **"shape"** — eager, TypeScript math, no backend. Runs on every edit.
- **"train"** — manual, WebSocket streaming, epoch-by-epoch progress. Auto-detects paradigm (standard, GAN, diffusion, autoregressive).
- **"test"** — evaluates on held-out test set (classification models only).
- **"infer"** — manual, uses trained weights stored in backend memory.

### Training paradigms (backend/training/)

The training loop is a plugin system. `train_graph()` auto-detects which paradigm to use:
- **standard** — single forward → loss → backward → update. Handles classification, reconstruction, VAEs.
- **gan** — alternating generator/discriminator updates with two optimizers.
- **diffusion** — noise-conditioned denoising with timestep injection.
- **autoregressive** — next-token prediction with 3D logit reshaping, perplexity metrics, text generation.

See `docs/training-plugins.md` for the full architecture documentation.

### Text preprocessing pipeline

The `ml.preprocessing.tokenizer` node sits between data and embedding. Three modes:
- **character** — each character is a token (default for Shakespeare, vocab ~65)
- **word** — split on whitespace/punctuation, frequency-based vocab (default for IMDb/AG News)
- **bpe** — Byte-Pair Encoding learned from the dataset (`backend/dataprep/bpe.py`). BPE merges are cached per (dataset, vocab_size) so training only happens once (~5s on Shakespeare).

During training, `build_training_context()` detects the tokenizer node's mode. If BPE, it learns merges from the corpus and creates a BPE-encoded dataset. During text generation, `text_generate.py` detects the mode and uses the matching encode/decode functions.

### Adding a new node type

1. **Frontend**: create file in `src/domain/nodes/<category>/`, export a `NodeDefinition`, add to that folder's `index.ts` array. If it's a new category (like `preprocessing/`), also import in `src/domain/index.ts`.
2. **Backend builder**: add builder function to `backend/engine/node_builders.py` (layers) or `backend/dataprep/data_loaders.py` (datasets). Loss nodes also need `LOSS_NODES` in `backend/engine/graph_builder/constants.py`.
3. **Backend viz**: create or edit a file in `backend/visualize/layers/` with the forward/backward viz function, then register it in `backend/visualize/node_viz.py`'s `FORWARD_VIZ` / `BACKWARD_VIZ` registries. Optional — default fallback provides basic shape-based viz.

Additional design docs live in `docs/`: `backend-architecture.md` (backend overview + "what happens when you press Train" walkthrough), `shape-inference.md`, `training-flow.md`, `training-plugins.md`, `visualization.md`, `multi-output-nodes.md`, `custom-blocks.md`, `copy-paste.md`, `undo-redo.md`.

### Node metadata convention

Nodes communicate display data through `metadata` on `ExecutionResult`:
- `shapes: { label: string; value: any[] | string }[]` — labeled shape rows on the node
- `outputShape: number[]` — fallback if no `shapes` array
- `paramCount: number` — shown as "X params" badge
- `paramBreakdown: string` — shown in inspector (e.g., "weights: 64x1x3x3 = 576 + bias: 64 = 640")
- `error: string` — red error box on the node
- `prediction: { predictedClass, confidence, probabilities }` — inference results
- `imagePixels: number[][]` — 2D pixel array for image preview
- `finalLoss / finalAccuracy` — training results on optimizer nodes

### Connection validation

`isValidConnection` in `useGraph.ts` checks: no self-connections, data type compatibility (via Layer 2), no duplicate connections to single-input ports. Rejection reasons logged to browser console.

### Model state

`modelTrained` / `modelStale` flags track whether inference is valid. Any graph mutation (add/remove node, add/remove edge, change property) sets `modelStale = true`. Training resets both flags.

## Key Design Decisions

- Data types are strings, not enums. Registered via `DataTypeRegistry`.
- `getPorts()` is a function of properties — enables dynamic ports (e.g., Concat with N inputs).
- Execution modes are registered, not hardcoded. Adding new modes doesn't touch engine code.
- `metadata` on ExecutionResult is unstructured — each node emits whatever makes sense for visualization.
- All React Flow imports use `import * as RF from '@xyflow/react'` namespace style for clarity.
- Frontend uses JSDoc (`/** */`) on all core interfaces and functions for hover documentation.
