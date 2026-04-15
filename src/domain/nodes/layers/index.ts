import type { NodeDefinition } from '../../../core/nodedef';
import { conv2dNode } from './conv2d';
import { linearNode } from './linear';
import { flattenNode } from './flatten';
import { maxPool2dNode } from './maxpool2d';
import { batchNorm2dNode } from './batchnorm2d';
import { dropoutNode } from './dropout';

export const layerNodes: NodeDefinition[] = [
  conv2dNode,
  linearNode,
  flattenNode,
  maxPool2dNode,
  batchNorm2dNode,
  dropoutNode,
];
