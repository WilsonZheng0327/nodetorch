import type { NodeDefinition } from '../../../core/nodedef';
import { addNode } from './add';
import { concatNode } from './concat';
import { reshapeNode } from './reshape';
import { permuteNode } from './permute';
import { sequencePoolNode } from './sequence-pool';
import { reparameterizeNode } from './reparameterize';

export const structuralNodes: NodeDefinition[] = [
  addNode,
  concatNode,
  reshapeNode,
  permuteNode,
  sequencePoolNode,
  reparameterizeNode,
];
