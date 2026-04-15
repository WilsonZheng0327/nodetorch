import type { NodeDefinition } from '../../../core/nodedef';
import { mnistNode } from './mnist';
import { cifar100Node } from './cifar100';

export const dataNodes: NodeDefinition[] = [
  mnistNode,
  cifar100Node,
];
