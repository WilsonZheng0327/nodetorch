# Copy/Paste

NodeTorch supports copying one or more selected nodes (with their internal edges) and pasting them as new, independent copies. The implementation lives in `src/ui/useGraph.ts` with keyboard bindings in `src/App.tsx`.

## Data structures

A single ref holds the clipboard contents:

```ts
const clipboard = useRef<{ nodes: any[]; edges: any[] } | null>(null);
```

When populated, the clipboard contains:

- **`nodes`**: An array of plain objects, each with:
  - `type` -- the node definition type string (e.g. `"ml.layers.conv2d"`)
  - `position` -- `{ x, y }` canvas coordinates at the time of copying
  - `properties` -- a shallow copy of the node's property values
  - `subgraph` (optional) -- a `SerializedGraphData` object if the node is a subgraph block

- **`edges`**: An array of index-based edge references, each with:
  - `sourceIdx` -- index into the `nodes` array for the source node
  - `sourcePort` -- port id on the source node
  - `targetIdx` -- index into the `nodes` array for the target node
  - `targetPort` -- port id on the target node

Edges are stored as indices rather than node IDs because the pasted copies will have new IDs. The indices provide a stable mapping from the original selection order to the newly created nodes.

## copySelected()

```ts
const copySelected = useCallback(() => {
  const currentGraph = getCurrentGraph();
  const selectedIds = new Set(
    rfNodes.filter((n) => n.selected).map((n) => n.id),
  );
  if (selectedIds.size === 0) return;

  const copiedNodes = Array.from(currentGraph.nodes.values())
    .filter((n) => selectedIds.has(n.id))
    .map((n) => ({
      type: n.type,
      position: { ...n.position },
      properties: { ...n.properties },
      subgraph: n.subgraph ? serializeGraphData(n.subgraph) : undefined,
    }));

  const copiedEdges = currentGraph.edges
    .filter((e) => selectedIds.has(e.source.nodeId) && selectedIds.has(e.target.nodeId))
    .map((e) => ({
      sourceIdx: Array.from(selectedIds).indexOf(e.source.nodeId),
      sourcePort: e.source.portId,
      targetIdx: Array.from(selectedIds).indexOf(e.target.nodeId),
      targetPort: e.target.portId,
    }));

  clipboard.current = { nodes: copiedNodes, edges: copiedEdges };
}, [getCurrentGraph, rfNodes]);
```

Step by step:

1. **Identify selected nodes.** The function reads `rfNodes` (React Flow state) and collects the IDs of all nodes with `selected: true`.
2. **Copy node data.** For each selected node, it captures the type, a spread copy of position and properties, and the serialized subgraph if present. Crucially, it does **not** store the node's `id` -- pasted copies get fresh IDs.
3. **Filter internal edges.** Only edges where **both** the source and target are in the selection are included. Edges connecting a selected node to an unselected node are discarded, since there would be no corresponding node to connect to after pasting.
4. **Convert edge node references to indices.** Each edge maps its source/target node IDs to indices in the `selectedIds` set (which preserves insertion order). These indices will be used during paste to look up the corresponding new node IDs.

## paste()

```ts
const paste = useCallback(async () => {
  if (!clipboard.current || clipboard.current.nodes.length === 0) return;

  snapshot();
  const currentGraph = getCurrentGraph();
  const OFFSET = 50;
  const newIds: string[] = [];

  for (const copied of clipboard.current.nodes) {
    const id = `${copied.type}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const node = createNode(id, copied.type, {
      x: copied.position.x + OFFSET,
      y: copied.position.y + OFFSET,
    }, { ...copied.properties });

    if (copied.subgraph) {
      node.subgraph = deserializeGraphData(copied.subgraph);
    }

    addNode(currentGraph, node);
    newIds.push(id);
  }

  for (const ce of clipboard.current.edges) {
    if (ce.sourceIdx >= 0 && ce.targetIdx >= 0 &&
        ce.sourceIdx < newIds.length && ce.targetIdx < newIds.length) {
      const edgeId = `e-${edgeCounter++}`;
      const edge = createEdge(edgeId, newIds[ce.sourceIdx], ce.sourcePort,
                              newIds[ce.targetIdx], ce.targetPort);
      try {
        addGraphEdge(currentGraph, edge);
      } catch { /* skip invalid edges */ }
    }
  }

  invalidateModel();
  await runShape();
}, [getCurrentGraph, snapshot, invalidateModel, runShape]);
```

Step by step:

1. **Guard.** If the clipboard is empty, return immediately.
2. **Snapshot.** Calls `snapshot()` so the paste operation is undoable via Ctrl+Z.
3. **Create new nodes.** For each copied node, a new ID is generated using `${type}-${Date.now()}-${random4chars}`. The position is offset by 50px in both x and y so the pasted nodes don't land exactly on top of the originals. Properties are shallow-copied. Subgraphs are fully deserialized from their serialized form, producing an independent copy.
4. **Build the ID mapping.** As each node is created, its new ID is appended to `newIds`. Since the iteration order matches the clipboard's `nodes` array, `newIds[i]` corresponds to `clipboard.nodes[i]`.
5. **Reconnect edges.** For each copied edge, the `sourceIdx` and `targetIdx` are used to look up the new node IDs from `newIds`. A new edge is created with a fresh `edgeId` (from the module-level `edgeCounter`) and added to the graph. If any edge creation fails (e.g., due to port mismatch), it is silently skipped.
6. **Invalidate and re-run.** `invalidateModel()` marks the trained model as stale. `runShape()` executes shape inference on the entire graph and syncs the result to React Flow.

## Keyboard shortcuts

Defined in `src/App.tsx` inside the same `keydown` event listener that handles undo/redo:

- **Ctrl+C** (or Cmd+C on macOS) calls `graph.copySelected()`
- **Ctrl+V** (or Cmd+V) calls `graph.paste()`

The handler ignores keypresses when focus is in an `<input>`, `<textarea>`, or `<select>` element.

```ts
if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
  e.preventDefault();
  graph.copySelected();
  return;
}
if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
  e.preventDefault();
  graph.paste();
  return;
}
```

## Worked example

Starting state: a Conv2d node connected to a ReLU node via an edge from `conv.out` to `relu.in`.

1. **User selects both nodes** (box select or Shift+click).
   - React Flow marks both nodes as `selected: true` in `rfNodes`.

2. **User presses Ctrl+C.**
   - `copySelected()` runs.
   - `selectedIds` = `{ "ml.layers.conv2d-1234", "ml.activations.relu-1235" }`.
   - `copiedNodes` captures both nodes' types, positions, and properties (no IDs stored).
   - The edge between them has both endpoints in `selectedIds`, so it is included.
   - The edge is stored as `{ sourceIdx: 0, sourcePort: "out", targetIdx: 1, targetPort: "in" }` (Conv2d is index 0, ReLU is index 1 in the iteration order).
   - `clipboard.current` = `{ nodes: [{type: "ml.layers.conv2d", ...}, {type: "ml.activations.relu", ...}], edges: [{sourceIdx: 0, sourcePort: "out", targetIdx: 1, targetPort: "in"}] }`.

3. **User presses Ctrl+V.**
   - `paste()` runs.
   - `snapshot()` saves the current graph state for undo.
   - Two new nodes are created:
     - `"ml.layers.conv2d-1713200000000-a3f2"` at `(original_x + 50, original_y + 50)` with the same properties as the original Conv2d.
     - `"ml.activations.relu-1713200000001-b7c4"` at `(original_x + 50, original_y + 50)` with the same properties as the original ReLU.
   - `newIds` = `["ml.layers.conv2d-1713200000000-a3f2", "ml.activations.relu-1713200000001-b7c4"]`.
   - The stored edge `{sourceIdx: 0, targetIdx: 1}` maps to `newIds[0]` -> `newIds[1]`, so a new edge is created from the new Conv2d's `"out"` port to the new ReLU's `"in"` port.
   - Shape inference runs. The canvas now shows four nodes: the original pair connected by an edge, and the pasted pair connected by their own edge, offset 50px down and to the right.

4. **User presses Ctrl+V again.**
   - Another pair appears, offset 50px from the *original* positions (not from the first paste). The clipboard stores the original positions and does not update them after pasting.

## Interaction with undo

Since `paste()` calls `snapshot()` before mutating the graph, pressing Ctrl+Z after a paste removes all pasted nodes and edges in one step, restoring the graph to its pre-paste state.

## Limitations

- **No cross-tab clipboard.** The clipboard is a ref local to the `useGraph` hook. Copying in one browser tab and pasting in another does not work. The system clipboard is not used.
- **Pasted nodes are not auto-selected.** After pasting, the newly created nodes do not automatically become selected. The user must manually select them to move them as a group.
- **Repeated paste uses the same offset.** Pasting multiple times places copies at the same offset from the original positions, so they stack on top of each other.
- **Shallow property copy.** Properties are spread-copied (`{ ...n.properties }`). If a property value is an object or array, the copy shares the same reference. In practice this is not an issue because node properties are typically primitive values (numbers, strings, booleans).
