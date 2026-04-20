import type { NodeDefinition } from '../../../core/nodedef';
import { crossEntropyLossNode } from './cross-entropy-loss';
import { mseLossNode } from './mse-loss';
import { vaeLossNode } from './vae-loss';

export const lossNodes: NodeDefinition[] = [
  crossEntropyLossNode,
  mseLossNode,
  vaeLossNode,
];
