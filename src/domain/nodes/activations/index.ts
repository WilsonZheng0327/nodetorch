import type { NodeDefinition } from '../../../core/nodedef';
import { reluNode } from './relu';
import { sigmoidNode } from './sigmoid';
import { softmaxNode } from './softmax';
import { geluNode } from './gelu';
import { tanhNode } from './tanh';
import { leakyReluNode } from './leaky-relu';

export const activationNodes: NodeDefinition[] = [
  reluNode,
  sigmoidNode,
  softmaxNode,
  geluNode,
  tanhNode,
  leakyReluNode,
];
