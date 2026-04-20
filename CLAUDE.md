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

# Backend (from repo root, uses .venv)
.venv/bin/python backend/main.py   # FastAPI server at http://localhost:8000
```

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

Backend mirrors the engine: `backend/graph_builder.py` does the same topological walk but with real PyTorch tensors.

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
- `backend/graph_builder.py` — graph→PyTorch execution, inference, model store
- `backend/node_builders.py` — per-node-type `nn.Module` builder functions
- `backend/node_viz.py` — per-node-type visualization registry (forward + backward step-through)
- `backend/data_loaders.py` — per-dataset loader functions
- `backend/training/` — training loop plugin system (standard, GAN, diffusion)
- `backend/step_through.py` — forward step-through orchestration (uses node_viz.py for viz)
- `backend/backprop_sim.py` — backward step-through + simple backprop animation
- `backend/denoise_viz.py` — diffusion denoising step-through visualization
- `backend/gan_generate.py` — GAN image generation on demand
- `backend/latent_viz.py` — VAE latent space grid visualization

### Execution modes

- **"shape"** — eager, TypeScript math, no backend. Runs on every edit.
- **"train"** — manual, WebSocket streaming, epoch-by-epoch progress. Auto-detects paradigm (standard, GAN, diffusion).
- **"test"** — evaluates on held-out test set (classification models only).
- **"infer"** — manual, uses trained weights stored in backend memory.

### Training paradigms (backend/training/)

The training loop is a plugin system. `train_graph()` auto-detects which paradigm to use:
- **standard** — single forward → loss → backward → update. Handles classification, reconstruction, VAEs.
- **gan** — alternating generator/discriminator updates with two optimizers.
- **diffusion** — noise-conditioned denoising with timestep injection.

See `docs/training-plugins.md` for the full architecture documentation.

### Adding a new node type

1. **Frontend**: create file in `src/domain/nodes/<category>/`, export a `NodeDefinition`, add to that folder's `index.ts` array. The domain `index.ts` doesn't change.
2. **Backend builder**: add builder function to `backend/node_builders.py` (layers) or `backend/data_loaders.py` (datasets). Loss nodes also need `LOSS_NODES` in `graph_builder.py`.
3. **Backend viz**: add forward/backward viz functions to `backend/node_viz.py` registries (`FORWARD_VIZ`, `BACKWARD_VIZ`). Optional — default fallback provides basic shape-based viz.

See `CONTRIBUTING.md` for full details with code examples and a per-node-type file change table.

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
