import type { NodeDefinition } from '../../../core/nodedef';
import { tokenizerNode } from './tokenizer';

export const preprocessingNodes: NodeDefinition[] = [
  tokenizerNode,
];
