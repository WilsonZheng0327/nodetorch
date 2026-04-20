// Layer 4: Node Registry
// Plugin-style registration. Defines what a node IS — its properties, ports, and executors.
// The UI uses list() for the node palette. The engine uses getExecutor() to run nodes.
// The graph core uses getPorts()/getProperties() to know a node's structure.

import type { Executor } from './engine';

// --- Port & Property Definitions ---

/**
 * Describes a port on a node — a connection point for edges.
 * Ports have a direction (input/output) and a data type (e.g. "tensor").
 */
export interface PortDefinition {
  /** Unique id within this node, e.g. "in", "out", "labels" */
  id: string;
  /** Display name, e.g. "Input", "Output" */
  name: string;
  /** "input" = receives data, "output" = sends data */
  direction: 'input' | 'output';
  /** References a DataTypeDefinition.id from Layer 2, e.g. "tensor" */
  dataType: string;
  /** If true, multiple edges can connect to this port */
  allowMultiple: boolean;
  /** If true, this port doesn't need a connection for the node to execute */
  optional: boolean;
  /** Value to use if nothing is connected and port is optional */
  defaultValue?: any;
}

/**
 * Describes the kind of value a property holds.
 * The UI uses this to pick the right widget (number input, toggle, dropdown, etc.).
 */
export type PropertyType =
  | { kind: 'number'; min?: number; max?: number; step?: number; integer?: boolean }
  | { kind: 'string' }
  | { kind: 'boolean' }
  | { kind: 'select'; options: { label: string; value: any }[] }
  | { kind: 'range'; min: number; max: number }
  | { kind: 'custom'; component: string };

/**
 * Describes an editable property on a node (e.g. Conv2d's kernelSize, outChannels).
 * The UI auto-generates editor widgets from these definitions.
 */
export interface PropertyDefinition {
  /** Unique id within this node, e.g. "kernelSize" */
  id: string;
  /** Display name, e.g. "Kernel Size" */
  name: string;
  /** What kind of value this is — determines the UI widget */
  type: PropertyType;
  /** Starting value when the node is created */
  defaultValue: any;
  /** Optional grouping label for organizing properties in the inspector */
  group?: string;
  /**
   * Controls whether this property is shown in the UI.
   * Takes all current properties and returns true/false.
   * e.g. "padding mode" only shows when padding > 0:
   * `visible: (props) => props.padding > 0`
   */
  visible?: (properties: Record<string, any>) => boolean;
  /**
   * What to recompute when this property changes:
   * - "ports" — recompute ports via getPorts() (may disconnect edges to removed ports)
   * - "execution" — mark dirty and re-run executors
   * - "both" — recompute ports AND re-run executors
   */
  affects?: 'ports' | 'execution' | 'both';
  /** Help text shown as a tooltip next to the property in the inspector. */
  help?: string;
}

// --- Node Definition ---

/**
 * The complete definition of a node type. Declares everything about how a node
 * looks, behaves, and executes. Layer 5 creates these and registers them.
 *
 * This is the DEFINITION (what Conv2d means), not an INSTANCE (a specific Conv2d
 * on the canvas). Instances live in Graph as NodeInstance.
 */
export interface NodeDefinition {
  /** Namespaced type string, e.g. "ml.layers.conv2d", "data.mnist" */
  type: string;
  /** Version number for serialization migration when the definition changes */
  version: number;
  /** Display name in the UI, e.g. "Conv2d" */
  displayName: string;
  /** Short description, e.g. "2D convolution layer" */
  description: string;
  /** Category path for organizing in the node palette, e.g. ["ML", "Layers", "Convolution"] */
  category: string[];
  /** Optional icon identifier */
  icon?: string;
  /** Optional header color override */
  color?: string;
  /** Educational explanation — shown in the inspector when a node is selected */
  learnMore?: string;

  /**
   * Returns the property definitions for this node type.
   * The UI uses this to build the property inspector.
   */
  getProperties(): PropertyDefinition[];
  /**
   * Returns the port definitions as a function of current properties.
   * This is a function (not a static list) so nodes can have dynamic ports —
   * e.g. Concat's input count depends on a property.
   */
  getPorts(properties: Record<string, any>): PortDefinition[];

  /**
   * Map of executorKey → Executor. A node only implements modes it supports.
   * e.g. { shape: ..., forward: ... } — no train executor means this node
   * is skipped during training.
   */
  executors: Record<string, Executor>;

  /** Optional validation beyond type checking. Returns warnings/errors for the UI. */
  validate?(properties: Record<string, any>, inputs: Record<string, any>): ValidationResult[];

  /** Called when a new instance of this node is created on the canvas */
  onCreate?(properties: Record<string, any>): void;
  /** Called when an instance is deleted — cleanup (e.g. free GPU resources) */
  onDestroy?(properties: Record<string, any>, state: any): void;

  /** Convert custom state (e.g. weight tensors) to a JSON-safe format for saving */
  serialize?(state: any): any;
  /**
   * Restore state from saved data. The version number enables migration —
   * if the definition changed in v2, deserialize(data, 1) converts v1 format to v2.
   */
  deserialize?(data: any, version: number): any;
}

/** A validation message returned by NodeDefinition.validate(). */
export interface ValidationResult {
  level: 'error' | 'warning' | 'info';
  message: string;
}

// --- Registry ---

/**
 * Stores all registered node definitions. Plugin-style — anyone can call register().
 * The UI reads list() for the node palette. The engine reads getExecutor() to run nodes.
 */
export class NodeRegistry {
  /** nodeType → NodeDefinition */
  private definitions = new Map<string, NodeDefinition>();

  /** Register a node definition. Throws if the type is already registered. */
  register(definition: NodeDefinition): void {
    if (this.definitions.has(definition.type)) {
      throw new Error(`Node type "${definition.type}" already registered`);
    }
    this.definitions.set(definition.type, definition);
  }

  /** Look up a node definition by type string. */
  get(type: string): NodeDefinition | undefined {
    return this.definitions.get(type);
  }

  /**
   * List all definitions, optionally filtered by category prefix.
   * e.g. list(["ML", "Layers"]) matches ["ML", "Layers", "Convolution"]
   */
  list(category?: string[]): NodeDefinition[] {
    const all = Array.from(this.definitions.values());
    if (!category) return all;

    return all.filter((def) =>
      category.every((cat, i) => def.category[i] === cat),
    );
  }

  /**
   * Find the executor for a given node type and executor key.
   * This is what gets passed to the engine as its ExecutorLookup function.
   * Returns undefined if the node type doesn't exist or doesn't have that executor.
   */
  getExecutor(nodeType: string, executorKey: string): Executor | undefined {
    return this.definitions.get(nodeType)?.executors[executorKey];
  }
}
