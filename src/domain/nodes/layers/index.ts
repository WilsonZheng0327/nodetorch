import type { NodeDefinition } from '../../../core/nodedef';
import { conv2dNode } from './conv2d';
import { conv1dNode } from './conv1d';
import { convTranspose2dNode } from './conv-transpose2d';
import { linearNode } from './linear';
import { flattenNode } from './flatten';
import { maxPool2dNode } from './maxpool2d';
import { maxPool1dNode } from './maxpool1d';
import { avgPool2dNode } from './avgpool2d';
import { adaptiveAvgPool2dNode } from './adaptive-avgpool2d';
import { batchNorm2dNode } from './batchnorm2d';
import { batchNorm1dNode } from './batchnorm1d';
import { groupNormNode } from './groupnorm';
import { instanceNorm2dNode } from './instancenorm2d';
import { dropoutNode } from './dropout';
import { dropout2dNode } from './dropout2d';
import { layerNormNode } from './layernorm';
import { embeddingNode } from './embedding';
import { multiHeadAttentionNode } from './multihead-attention';
import { attentionNode } from './attention';
import { lstmNode } from './lstm';
import { gruNode } from './gru';
import { rnnNode } from './rnn';
import { upsampleNode } from './upsample';
import { pretrainedResnet18Node } from './pretrained-resnet18';

export const layerNodes: NodeDefinition[] = [
  conv2dNode,
  conv1dNode,
  convTranspose2dNode,
  linearNode,
  flattenNode,
  maxPool2dNode,
  maxPool1dNode,
  avgPool2dNode,
  adaptiveAvgPool2dNode,
  batchNorm2dNode,
  batchNorm1dNode,
  groupNormNode,
  instanceNorm2dNode,
  dropoutNode,
  dropout2dNode,
  layerNormNode,
  embeddingNode,
  multiHeadAttentionNode,
  attentionNode,
  lstmNode,
  gruNode,
  rnnNode,
  upsampleNode,
  pretrainedResnet18Node,
];
