import type { NodeDefinition } from '../../../core/nodedef';
import { noiseSchedulerNode } from './noise-scheduler';

export const diffusionNodes: NodeDefinition[] = [
  noiseSchedulerNode,
];
