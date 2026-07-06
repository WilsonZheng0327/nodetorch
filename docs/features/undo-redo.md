# Undo/Redo

NodeTorch uses a snapshot-based undo/redo system. Before every mutation that changes the graph structure or properties, the entire graph state is serialized to a JSON string and pushed onto an undo stack. Undoing pops that string, deserializes it back into a live `Graph` object, re-runs shape inference, and syncs the result to React Flow.

## Data structures

All state lives in `src/ui/useGraph.ts` as refs inside the `useGraph` hook:

```ts
const undoStack = useRef<string[]>([]);   // past states (JSON strings)
const redoStack = useRef<string[]>([]);   // future states (JSON strings)
const isUndoRedo = useRef(false);         // guard flag
```

Both stacks store serialized `SerializedGraphData` objects (not the full `SerializedGraph` wrapper). Each entry is a single `JSON.stringify(serializeGraphData(graphRef.current))` call, capturing every node (id, type, position, properties, and nested subgraphs) and every edge.

The undo stack is capped at 50 entries. When it exceeds that limit, the oldest entry is shifted off the front.

## snapshot()

`snapshot()` is the function that captures the current graph state before a mutation occurs:

```ts
const snapshot = useCallback(() => {
  if (isUndoRedo.current) return;        // don't record while restoring
  undoStack.current.push(JSON.stringify(serializeGraphData(graphRef.current)));
  redoStack.current = [];                // any new edit clears the redo stack
  if (undoStack.current.length > 50) undoStack.current.shift();
}, []);
```

Key behaviors:
- If `isUndoRedo` is true, `snapshot()` returns immediately. This prevents undo/redo operations from recording themselves as new edits.
- Every call clears `redoStack`. Once the user makes a new edit after undoing, the redo history is gone (standard undo/redo semantics).

### Where snapshot() is called

`snapshot()` is called at the start of every graph-mutating action, before the actual mutation:

| Action | Function | What it does after snapshot |
|---|---|---|
| Add a node | `addNodeToGraph()` | `addNode(currentGraph, node)` |
| Remove a node | `removeNodeFromGraph()` | `removeNode(getCurrentGraph(), nodeId)` |
| Connect two nodes | `connectNodes()` | `addGraphEdge(currentGraph, edge)` |
| Change a property | `updateProperty()` | `setProperty(getCurrentGraph(), nodeId, key, value)` |
| Delete a node via React Flow | `onNodesChange()` | `g.nodes.delete(change.id)` + edge cleanup |
| Delete an edge via React Flow | `onEdgesChange()` | `g.edges.splice(idx, 1)` |
| Paste nodes | `paste()` | Creates new nodes and reconnects edges |

Note that `onNodesChange` and `onEdgesChange` only call `snapshot()` for `'remove'` type changes. Position-only drags do not create undo entries.

## undo()

```ts
const undo = useCallback(async () => {
  if (undoStack.current.length === 0) return;
  // 1. Save current state to redo stack
  redoStack.current.push(JSON.stringify(serializeGraphData(graphRef.current)));
  // 2. Pop previous state from undo stack
  const prev = undoStack.current.pop()!;
  // 3. Set guard flag
  isUndoRedo.current = true;
  // 4. Deserialize into a fresh Graph object
  graphRef.current = deserializeGraphData(JSON.parse(prev));
  // 5. Re-run shape inference (which also calls syncToRF)
  await runShape();
  // 6. Clear guard flag
  isUndoRedo.current = false;
}, [runShape]);
```

The `isUndoRedo` flag is set to `true` before any graph mutation and cleared after `runShape()` completes. `runShape()` calls `domain.engine.execute(...)` (shape mode, pure TypeScript math) and then `syncToRF()`, which converts the graph to React Flow nodes/edges and updates React state.

## redo()

`redo()` is the mirror image: it pushes the current state onto the undo stack, pops from the redo stack, deserializes, and re-runs shape inference under the same `isUndoRedo` guard.

```ts
const redo = useCallback(async () => {
  if (redoStack.current.length === 0) return;
  undoStack.current.push(JSON.stringify(serializeGraphData(graphRef.current)));
  const next = redoStack.current.pop()!;
  isUndoRedo.current = true;
  graphRef.current = deserializeGraphData(JSON.parse(next));
  await runShape();
  isUndoRedo.current = false;
}, [runShape]);
```

## Keyboard shortcuts

Defined in `src/App.tsx` inside a `keydown` event listener:

- **Ctrl+Z** (or Cmd+Z on macOS) calls `graph.undo()`
- **Ctrl+Shift+Z** (or Cmd+Shift+Z) calls `graph.redo()`

The handler ignores keypresses when focus is in an `<input>`, `<textarea>`, or `<select>` element, so typing in the property inspector does not trigger undo.

```ts
if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
  e.preventDefault();
  graph.undo();
  return;
}
if ((e.ctrlKey || e.metaKey) && (e.key === 'Z' || (e.key === 'z' && e.shiftKey))) {
  e.preventDefault();
  graph.redo();
  return;
}
```

## Serialization details

`serializeGraphData()` converts a `Graph` (which uses a `Map` for nodes) into a plain JSON-serializable object. Subgraphs are serialized recursively. `deserializeGraphData()` reconstructs the `Graph` by calling `createGraph()`, `createNode()`, `addNode()`, and `addEdge()` from `src/core/graph.ts`.

Important: serialization captures node id, type, position, properties, and subgraph data. It does **not** capture `lastResult`, `state`, or `dirty` flags. After deserialization, all nodes start with `dirty: true`, so `runShape()` re-executes shape inference on every node to repopulate `lastResult`.

## Worked example

Starting state: empty graph.

1. **User drags a Conv2d onto the canvas.**
   - `addNodeToGraph("ml.layers.conv2d", {x:300, y:100})` is called.
   - `snapshot()` runs first: serializes the empty graph and pushes it onto `undoStack`. (`undoStack = ["(empty graph)"]`, `redoStack = []`)
   - The Conv2d node is added to the graph. Shape inference runs.

2. **User changes the Conv2d's `outChannels` from 64 to 128.**
   - `updateProperty(convId, "outChannels", 128)` is called.
   - `snapshot()` runs: serializes the graph-with-Conv2d(outChannels=64) and pushes. (`undoStack = ["(empty graph)", "(graph with Conv2d, outChannels=64)"]`, `redoStack = []`)
   - The property is updated. Shape inference re-runs.

3. **User presses Ctrl+Z (first undo).**
   - `undo()` runs.
   - Current state (Conv2d with outChannels=128) is pushed onto `redoStack`.
   - The top of `undoStack` is popped: "(graph with Conv2d, outChannels=64)".
   - `isUndoRedo` is set to `true`.
   - The JSON is parsed and `deserializeGraphData` rebuilds the graph with the Conv2d at outChannels=64.
   - `runShape()` executes shape inference on the restored graph, then `syncToRF()` updates React Flow.
   - `isUndoRedo` is set to `false`.
   - Result: (`undoStack = ["(empty graph)"]`, `redoStack = ["(graph with Conv2d, outChannels=128)"]`)
   - The canvas shows Conv2d with outChannels=64.

4. **User presses Ctrl+Z again (second undo).**
   - `undo()` runs.
   - Current state (Conv2d with outChannels=64) is pushed onto `redoStack`.
   - The top of `undoStack` is popped: "(empty graph)".
   - The graph is restored to empty. Shape inference runs (no nodes, nothing to compute).
   - Result: (`undoStack = []`, `redoStack = ["(graph with Conv2d, outChannels=128)", "(graph with Conv2d, outChannels=64)"]`)
   - The canvas is empty again.

At this point, pressing Ctrl+Z does nothing (undo stack is empty). Pressing Ctrl+Shift+Z would redo, restoring the Conv2d with outChannels=64 first, then outChannels=128.

## Limitations

- **No partial undo.** Every snapshot is the entire graph. This is simple and correct but uses more memory for large graphs.
- **50-entry cap.** Older history is silently discarded.
- **Node position drags are not undoable.** Only structural changes (add/remove/connect) and property edits are captured.
- **Redo is lost on new edits.** Any mutation after an undo clears the redo stack.
