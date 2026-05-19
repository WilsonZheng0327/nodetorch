import type { NodeDefinition } from '../../../core/nodedef';
import { tokenizerCharNode } from './tokenizer-char';
import { tokenizerWordNode } from './tokenizer-word';
import { tokenizerBpeNode } from './tokenizer-bpe';

export const preprocessingNodes: NodeDefinition[] = [
  tokenizerCharNode,
  tokenizerWordNode,
  tokenizerBpeNode,
];
