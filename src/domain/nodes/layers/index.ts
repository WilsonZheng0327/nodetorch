import type { NodeDefinition } from '../../../core/nodedef';
import { conv2dNode } from './conv2d';
import { conv1dNode } from './conv1d';
import { linearNode } from './linear';
import { flattenNode } from './flatten';
import { maxPool2dNode } from './maxpool2d';
import { avgPool2dNode } from './avgpool2d';
import { adaptiveAvgPool2dNode } from './adaptive-avgpool2d';
import { batchNorm2dNode } from './batchnorm2d';
import { batchNorm1dNode } from './batchnorm1d';
import { dropoutNode } from './dropout';
import { layerNormNode } from './layernorm';
import { embeddingNode } from './embedding';
import { multiHeadAttentionNode } from './multihead-attention';
import { attentionNode } from './attention';

export const layerNodes: NodeDefinition[] = [
  conv2dNode,
  conv1dNode,
  linearNode,
  flattenNode,
  maxPool2dNode,
  avgPool2dNode,
  adaptiveAvgPool2dNode,
  batchNorm2dNode,
  batchNorm1dNode,
  dropoutNode,
  layerNormNode,
  embeddingNode,
  multiHeadAttentionNode,
  attentionNode,
];
