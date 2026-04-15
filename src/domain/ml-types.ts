// Register ML-specific data types into Layer 2's DataTypeRegistry.

import { DataTypeRegistry } from '../core/datatypes';

export function registerMLTypes(registry: DataTypeRegistry): void {
  registry.register({
    id: 'tensor',
    label: 'Tensor',
    color: '#3b82f6',  // blue
  });

  registry.register({
    id: 'scalar',
    label: 'Scalar',
    color: '#10b981',  // green
    compatibleWith: ['tensor'],  // a scalar is a 0-d tensor
  });

  registry.register({
    id: 'dataset',
    label: 'Dataset',
    color: '#f59e0b',  // orange
  });
}
