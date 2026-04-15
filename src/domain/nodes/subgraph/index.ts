import type { NodeDefinition } from '../../../core/nodedef';
import { graphInputNode } from './graph-input';
import { graphOutputNode } from './graph-output';
import { subgraphNode } from './subgraph';

export const subgraphNodes: NodeDefinition[] = [
  graphInputNode,
  graphOutputNode,
  subgraphNode,
];
