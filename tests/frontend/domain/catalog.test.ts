// Tests for the agent node catalog builder (src/domain/catalog.ts).
// The catalog is the single source of truth sent to the AI agent, so it must
// cover every registered node with its properties and ports.

import { describe, it, expect } from 'vitest';

import { initDomain } from '../../../src/domain';
import { buildNodeCatalog } from '../../../src/domain/catalog';

describe('buildNodeCatalog', () => {
  const domain = initDomain();
  const catalog = buildNodeCatalog(domain);

  it('covers every registered node type', () => {
    expect(catalog.length).toBe(domain.nodeRegistry.list().length);
    expect(catalog.length).toBeGreaterThan(10);
  });

  it('each entry has type, properties, ports, and modes', () => {
    for (const node of catalog) {
      expect(typeof node.type).toBe('string');
      expect(typeof node.displayName).toBe('string');
      expect(Array.isArray(node.category)).toBe(true);
      expect(Array.isArray(node.properties)).toBe(true);
      expect(Array.isArray(node.ports)).toBe(true);
      expect(Array.isArray(node.modes)).toBe(true);
    }
  });

  it('describes conv2d properties and ports', () => {
    const conv = catalog.find((n) => n.type === 'ml.layers.conv2d');
    expect(conv).toBeTruthy();
    expect(conv!.properties.some((p) => p.id === 'outChannels')).toBe(true);
    expect(conv!.ports.some((p) => p.direction === 'input')).toBe(true);
    expect(conv!.ports.some((p) => p.direction === 'output')).toBe(true);
  });

  it('captures select-property options (optimizer scheduler)', () => {
    const adam = catalog.find((n) => n.type === 'ml.optimizers.adam');
    expect(adam).toBeTruthy();
    const selectProp = adam!.properties.find((p) => p.kind === 'select');
    expect(selectProp?.options && selectProp.options.length).toBeGreaterThan(0);
  });
});
