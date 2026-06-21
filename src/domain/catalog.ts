// Node catalog — a machine-readable description of every node type, built from
// the registry (the single source of truth). Sent to the AI agent so it knows
// exactly what nodes exist and how to configure them. New nodes appear in the
// catalog automatically; nothing is duplicated on the backend.

import type { NodeRegistry, PropertyType } from '../core/nodedef';

export interface CatalogProperty {
  id: string;
  name: string;
  kind: PropertyType['kind'];
  default?: unknown;
  min?: number;
  max?: number;
  integer?: boolean;
  options?: { label: string; value: unknown }[];
}

export interface CatalogPort {
  id: string;
  direction: 'input' | 'output';
  dataType: string;
  allowMultiple: boolean;
  optional: boolean;
}

export interface CatalogNode {
  type: string;
  displayName: string;
  category: string[];
  description?: string;
  learnMore?: string;
  properties: CatalogProperty[];
  ports: CatalogPort[];
  modes: string[];
}

export type NodeCatalog = CatalogNode[];

function describeProperty(prop: {
  id: string;
  name: string;
  type: PropertyType;
  defaultValue: unknown;
}): CatalogProperty {
  const t = prop.type;
  const out: CatalogProperty = {
    id: prop.id,
    name: prop.name,
    kind: t.kind,
    default: prop.defaultValue,
  };
  if (t.kind === 'number') {
    if (t.min !== undefined) out.min = t.min;
    if (t.max !== undefined) out.max = t.max;
    if (t.integer) out.integer = true;
  } else if (t.kind === 'range') {
    out.min = t.min;
    out.max = t.max;
  } else if (t.kind === 'select') {
    out.options = t.options;
  }
  return out;
}

/**
 * Build the node catalog from the registry. Ports are computed from each node's
 * default properties (ports can depend on properties, so this is the catalog's
 * baseline view).
 */
export function buildNodeCatalog(domain: { nodeRegistry: NodeRegistry }): NodeCatalog {
  return domain.nodeRegistry.list().map((def): CatalogNode => {
    const properties = def.getProperties().map(describeProperty);
    const defaultProps: Record<string, unknown> = {};
    for (const p of def.getProperties()) defaultProps[p.id] = p.defaultValue;

    const ports = def.getPorts(defaultProps).map(
      (p): CatalogPort => ({
        id: p.id,
        direction: p.direction,
        dataType: p.dataType,
        allowMultiple: p.allowMultiple,
        optional: p.optional,
      }),
    );

    return {
      type: def.type,
      displayName: def.displayName,
      category: def.category,
      description: def.description,
      learnMore: def.learnMore,
      properties,
      ports,
      modes: Object.keys(def.executors),
    };
  });
}
