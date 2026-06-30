// Shared React contexts for the node UI.
//
// These live in their own module (rather than in EngineNode.tsx) so that
// EngineNode.tsx exports only its component. A file that exports both a
// component and non-component values (like these contexts) disables React Fast
// Refresh for that file — every edit forces a full page reload, losing the
// on-canvas graph. Keeping the contexts here keeps EngineNode hot-swappable.

import { createContext } from 'react';
import type { DomainContext } from '../domain';
import type { VizSnapshot } from './VizPanel';

// The domain context is provided at the top of the React tree
// so every node component can look up definitions.
export const DomainCtx = createContext<DomainContext | null>(null);

// Callback context for actions that nodes can trigger
export const GraphActionsCtx = createContext<{
  removeNode: (nodeId: string) => void;
} | null>(null);

// Visualization context for live training snapshots and pinned panels
export const VizCtx = createContext<{
  pinnedVizNodes: Set<string>;
  toggleVizPin: (nodeId: string) => void;
  liveSnapshots: Record<string, VizSnapshot>;
} | null>(null);

// Backprop animation context — map of nodeId to { delayMs, intensity }
export const BackpropCtx = createContext<Record<string, { delayMs: number; intensity: number }> | null>(null);
