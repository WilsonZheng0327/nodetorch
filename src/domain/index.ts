// Layer 5 bootstrap: registers all ML-specific content into the generic engine.
// Call initDomain() once at startup.

import { DataTypeRegistry } from '../core/datatypes';
import { ExecutionEngine } from '../core/engine';
import { NodeRegistry } from '../core/nodedef';
import { registerMLTypes } from './ml-types';
import { registerMLModes } from './ml-modes';

// Each folder exports an array of NodeDefinitions
import { dataNodes } from './nodes/data';
import { layerNodes } from './nodes/layers';
import { activationNodes } from './nodes/activations';
import { lossNodes } from './nodes/loss';
import { optimizerNodes } from './nodes/optimizers';
import { structuralNodes } from './nodes/structural';
import { subgraphNodes } from './nodes/subgraph';

const allNodes = [
  ...dataNodes,
  ...layerNodes,
  ...activationNodes,
  ...lossNodes,
  ...optimizerNodes,
  ...structuralNodes,
  ...subgraphNodes,
];

export interface DomainContext {
  typeRegistry: DataTypeRegistry;
  engine: ExecutionEngine;
  nodeRegistry: NodeRegistry;
}

export function initDomain(): DomainContext {
  const typeRegistry = new DataTypeRegistry();
  const engine = new ExecutionEngine();
  const nodeRegistry = new NodeRegistry();

  registerMLTypes(typeRegistry);
  registerMLModes(engine);

  for (const node of allNodes) {
    nodeRegistry.register(node);
  }

  return { typeRegistry, engine, nodeRegistry };
}
