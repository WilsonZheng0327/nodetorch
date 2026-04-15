import type { NodeDefinition } from '../../../core/nodedef';
import { addNode } from './add';
import { concatNode } from './concat';
import { reshapeNode } from './reshape';
import { permuteNode } from './permute';

export const structuralNodes: NodeDefinition[] = [
  addNode,
  concatNode,
  reshapeNode,
  permuteNode,
];
