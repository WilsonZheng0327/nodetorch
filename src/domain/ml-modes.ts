// Register ML-specific execution modes into Layer 3's ExecutionEngine.

import { ExecutionEngine } from '../core/engine';

// id vs executorKey:
// - id identifies the mode (used when calling engine.execute(graph, "shape", ...))
// - executorKey is which key to look up in a node's executors map
// They're the same today. They'd diverge if two modes share the same executor —
// e.g., a "debug" mode with id: "debug" but executorKey: "forward" (same executor,
// different propagation/caching settings).
export function registerMLModes(engine: ExecutionEngine): void {
  engine.registerMode({
    id: 'shape',
    label: 'Shape Inference',
    propagation: 'eager',
    caching: true,
    executorKey: 'shape',
  });

  engine.registerMode({
    id: 'forward',
    label: 'Forward Pass',
    propagation: 'manual',
    caching: true,
    executorKey: 'forward',
  });

  engine.registerMode({
    id: 'train',
    label: 'Training',
    propagation: 'manual',
    caching: false,
    executorKey: 'train',
  });
}
