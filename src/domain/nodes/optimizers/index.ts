import type { NodeDefinition } from '../../../core/nodedef';
import { sgdNode } from './sgd';
import { adamNode } from './adam';
import { adamwNode } from './adamw';

export const optimizerNodes: NodeDefinition[] = [
  sgdNode,
  adamNode,
  adamwNode,
];
