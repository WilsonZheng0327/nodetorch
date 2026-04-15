import type { NodeDefinition } from '../../../core/nodedef';
import { conv2dNode } from './conv2d';
import { linearNode } from './linear';
import { flattenNode } from './flatten';
import { maxPool2dNode } from './maxpool2d';
import { batchNorm2dNode } from './batchnorm2d';
import { dropoutNode } from './dropout';
import { layerNormNode } from './layernorm';
import { embeddingNode } from './embedding';
import { multiHeadAttentionNode } from './multihead-attention';

export const layerNodes: NodeDefinition[] = [
  conv2dNode,
  linearNode,
  flattenNode,
  maxPool2dNode,
  batchNorm2dNode,
  dropoutNode,
  layerNormNode,
  embeddingNode,
  multiHeadAttentionNode,
];
