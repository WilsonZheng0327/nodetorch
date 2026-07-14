// Tests for src/core/datatypes.ts — Layer 2. The registry decides which port
// connections are type-compatible (same type, or an explicit one-way conversion)
// and supplies wire colors. It knows nothing about specific ML types itself.

import { describe, it, expect } from 'vitest';
import { DataTypeRegistry } from '../../../src/core/datatypes';

function make() {
  const r = new DataTypeRegistry();
  r.register({ id: 'tensor', label: 'Tensor', color: '#3b82f6' });
  r.register({ id: 'scalar', label: 'Scalar', color: '#10b981', compatibleWith: ['tensor'] });
  return r;
}

describe('DataTypeRegistry — registration', () => {
  it('stores and returns registered definitions', () => {
    const r = make();
    expect(r.get('tensor')?.label).toBe('Tensor');
  });

  it('returns undefined for unknown ids', () => {
    expect(make().get('nope')).toBeUndefined();
  });

  it('throws on duplicate registration', () => {
    const r = make();
    expect(() => r.register({ id: 'tensor', label: 'Dup', color: '#000' })).toThrow();
  });
});

describe('DataTypeRegistry — isCompatible', () => {
  it('accepts identical types', () => {
    expect(make().isCompatible('tensor', 'tensor')).toBe(true);
  });

  it('accepts a declared one-way conversion (scalar → tensor)', () => {
    expect(make().isCompatible('scalar', 'tensor')).toBe(true);
  });

  it('is directional: the reverse conversion is NOT implied', () => {
    // tensor does not declare compatibility with scalar.
    expect(make().isCompatible('tensor', 'scalar')).toBe(false);
  });

  it('rejects unrelated types', () => {
    const r = make();
    r.register({ id: 'dataset', label: 'Dataset', color: '#f59e0b' });
    expect(r.isCompatible('tensor', 'dataset')).toBe(false);
  });

  it('treats an unknown output type as incompatible (no crash)', () => {
    expect(make().isCompatible('mystery', 'tensor')).toBe(false);
  });
});

describe('DataTypeRegistry — getColor', () => {
  it('returns the registered wire color', () => {
    expect(make().getColor('scalar')).toBe('#10b981');
  });

  it('falls back to gray for unknown types', () => {
    expect(make().getColor('unknown')).toBe('#6b7280');
  });
});
