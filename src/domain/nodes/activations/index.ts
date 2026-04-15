import type { NodeDefinition } from '../../../core/nodedef';
import { reluNode } from './relu';
import { sigmoidNode } from './sigmoid';
import { softmaxNode } from './softmax';

export const activationNodes: NodeDefinition[] = [
  reluNode,
  sigmoidNode,
  softmaxNode,
];
