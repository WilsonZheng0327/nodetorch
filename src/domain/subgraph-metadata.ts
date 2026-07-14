// Layer 5: subgraph (composite block) metadata enrichment.
// The engine (Layer 3) builds only generic shape metadata for a block. This adds
// the ML/presentation fields — an aggregated parameter count and a per-layer
// breakdown string — that the inspector shows. Registered on the engine via
// engine.setSubgraphMetadataBuilder() in initDomain(), keeping this knowledge out
// of the ML-agnostic core.

import type { SubgraphMetadataBuilder } from '../core/engine';

/**
 * Sum the param counts of a block's inner nodes and format a breakdown, e.g.
 * "conv2d: 576 + linear: 128 = 704". Reads the generic `paramCount` each inner
 * node's executor already reported in its result metadata.
 */
export const buildSubgraphMetadata: SubgraphMetadataBuilder = (subgraph, baseMetadata) => {
  let totalParams = 0;
  const paramParts: string[] = [];
  for (const [, innerNode] of subgraph.nodes) {
    const pc = innerNode.lastResult?.metadata?.paramCount;
    if (pc) {
      totalParams += pc;
      const shortName = innerNode.type.split('.').pop() ?? innerNode.type;
      paramParts.push(`${shortName}: ${pc.toLocaleString()}`);
    }
  }

  return {
    ...baseMetadata,
    paramCount: totalParams || undefined,
    paramBreakdown:
      paramParts.length > 0
        ? paramParts.join(' + ') + ` = ${totalParams.toLocaleString()}`
        : undefined,
  };
};
