import type { NodeDefinition } from '../../../core/nodedef';
import { sgdNode } from './sgd';
import { adamNode } from './adam';

export const optimizerNodes: NodeDefinition[] = [
  sgdNode,
  adamNode,
];
