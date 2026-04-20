import type { NodeDefinition } from '../../../core/nodedef';
import { crossEntropyLossNode } from './cross-entropy-loss';
import { mseLossNode } from './mse-loss';
import { vaeLossNode } from './vae-loss';
import { ganLossNode } from './gan-loss';

export const lossNodes: NodeDefinition[] = [
  crossEntropyLossNode,
  mseLossNode,
  vaeLossNode,
  ganLossNode,
];
