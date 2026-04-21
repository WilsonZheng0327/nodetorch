import type { NodeDefinition } from '../../../core/nodedef';
import { mnistNode } from './mnist';
import { cifar100Node } from './cifar100';
import { cifar10Node } from './cifar10';
import { fashionMnistNode } from './fashion-mnist';
import { imdbNode } from './imdb';
import { agNewsNode } from './ag-news';
import { tinyShakespeareNode } from './tiny-shakespeare';

export const dataNodes: NodeDefinition[] = [
  mnistNode,
  cifar10Node,
  cifar100Node,
  fashionMnistNode,
  imdbNode,
  agNewsNode,
  tinyShakespeareNode,
];
