import type { NodeDefinition } from '../../../core/nodedef';
import { noiseSchedulerNode } from './noise-scheduler';
import { timestepEmbedNode } from './timestep-embed';

export const diffusionNodes: NodeDefinition[] = [
  noiseSchedulerNode,
  timestepEmbedNode,
];
