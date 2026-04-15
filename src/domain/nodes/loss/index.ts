import type { NodeDefinition } from '../../../core/nodedef';
import { crossEntropyLossNode } from './cross-entropy-loss';
import { mseLossNode } from './mse-loss';

export const lossNodes: NodeDefinition[] = [
  crossEntropyLossNode,
  mseLossNode,
];
