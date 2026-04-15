// Layer 2: Type System
// Extensible data type registry. Determines which ports can connect.
// Does NOT know about specific types like "tensor" — those are registered by Layer 5.

/**
 * Describes a data type that can flow through ports (e.g. tensor, scalar, dataset).
 * Registered by Layer 5 — the core doesn't know about specific types.
 */
export interface DataTypeDefinition {
  /** Unique string identifier, e.g. "tensor", "scalar", "dataset" */
  id: string;
  /** Display name shown in the UI */
  label: string;
  /** Wire color in the UI, e.g. "#3b82f6" for blue */
  color: string;
  /**
   * Types this can implicitly convert to.
   * e.g. scalar declares ["tensor"] — a scalar output can connect to a tensor input
   * without needing an explicit converter node.
   */
  compatibleWith?: string[];
}

/**
 * Holds all registered data types and answers "can this output connect to that input?"
 * Layer 5 populates it with register() calls. Layer 1 delegates connection validation here.
 */
export class DataTypeRegistry {
  /** typeId → definition */
  private types = new Map<string, DataTypeDefinition>();

  /** Register a new data type. Throws if already registered. */
  register(def: DataTypeDefinition): void {
    if (this.types.has(def.id)) {
      throw new Error(`Data type "${def.id}" already registered`);
    }
    this.types.set(def.id, def);
  }

  /** Look up a data type definition by id. */
  get(id: string): DataTypeDefinition | undefined {
    return this.types.get(id);
  }

  /**
   * Can an output of type `outputType` connect to an input of type `inputType`?
   * Returns true if they're the same type, or if outputType declares compatibility.
   */
  isCompatible(outputType: string, inputType: string): boolean {
    if (outputType === inputType) return true;

    const def = this.types.get(outputType);
    if (def?.compatibleWith?.includes(inputType)) return true;

    return false;
  }

  /** Get the wire color for a data type. Falls back to gray if not found. */
  getColor(id: string): string {
    return this.types.get(id)?.color ?? '#6b7280';
  }
}
