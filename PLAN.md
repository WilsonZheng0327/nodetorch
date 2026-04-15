# NodeTorch — Development Plan

A node-based visual tool for building, inspecting, and understanding ML models. Educational and open-source. Built with React + TypeScript frontend and Python (FastAPI + PyTorch) backend.

Build order follows the 6-layer architecture — each layer only depends on layers below it.

## Layer 1: Graph Core

The foundation. Nodes, ports, edges, and graph topology. No knowledge of ML, types, or execution.

- [ ] `Graph` — holds nodes (Map<string, NodeInstance>) and edges array
- [ ] `NodeInstance` — id, type (string), position, properties, state, dirty flag, lastResult, errors
- [ ] `Edge` — source (nodeId + portId) → target (nodeId + portId)
- [ ] Add/remove nodes and edges
- [ ] Topological sort — determine execution order from edges (DAG sort)
- [ ] Dirty tracking — when a node's property changes, mark it + all downstream nodes dirty
- [ ] Port management — `getPorts()` is a function of properties, recompute on property change, disconnect edges to removed ports
- [ ] Serialization — save/load graphs to JSON (string-based types make this generic)
- [ ] Subgraph support — `NodeInstance.subgraph` holds an inner `Graph`
- [ ] Tests for topo sort, dirty propagation, add/remove, serialization round-trip

## Layer 2: Type System

Extensible data type registry. Determines which ports can connect.

- [ ] `DataTypeDefinition` — id (string), label, color, compatibleWith list
- [ ] `DataTypeRegistry` — register(), isCompatible(), getColor()
- [ ] Connection validation — output type must be compatible with input type
- [ ] Wire Layer 2 into Layer 1 — graph core delegates type checks to registry
- [ ] Tests for compatibility rules, implicit conversions

## Layer 3: Execution Engine

Runs the graph in different modes. Doesn't know what the modes do — just how to propagate.

- [ ] `ExecutionModeDefinition` — id, label, propagation (eager/lazy/manual), caching, executorKey
- [ ] `ExecutionEngine` — registerMode(), execute(graph, mode)
- [ ] Execution flow: get topo order → skip clean nodes → gather inputs → call executor → store result → mark clean
- [ ] Eager propagation — auto-execute on dirty nodes (for shape inference)
- [ ] Manual propagation — execute only when triggered (for forward/train)
- [ ] Caching — skip execution if node is clean and inputs unchanged
- [ ] Tests for execution order, caching, eager vs manual

## Layer 4: Node Registry

Plugin-style registration. The interface between node definitions and the rest of the system.

- [ ] `NodeDefinition` — type, version, displayName, description, category, getProperties(), getPorts(), executors, validate, lifecycle hooks, serialize/deserialize, renderConfig
- [ ] `PropertyDefinition` — id, name, type (number/string/boolean/select/range/custom), defaultValue, group, visible(), affects (ports/execution/both)
- [ ] `PortDefinition` — id, name, direction, dataType, allowMultiple, optional, defaultValue
- [ ] `Executor` interface — execute(context) → ExecutionResult (outputs, state, metadata)
- [ ] `NodeRegistry` — register(), get(), list(category?)
- [ ] Wire Layer 4 into Layer 1 — graph core looks up getPorts/getProperties from registry
- [ ] Wire Layer 4 into Layer 3 — execution engine looks up executors from registry
- [ ] Tests for register/get/list, dynamic ports from properties

## Layer 5: ML Domain

Populates the generic engine with ML-specific content. Everything here is just register() calls.

### Data types
- [ ] Register tensor (blue), scalar (green), dataset (orange)
- [ ] Compatibility: scalar → tensor

### Execution modes
- [ ] Register "shape" — eager, cached (real-time shape feedback)
- [ ] Register "forward" — manual, cached (actual data, delegates to Python backend)
- [ ] Register "train" — manual, uncached (iterative forward+backward)

### Node definitions — Data
- [ ] MNIST — shape executor returns [1, 1, 28, 28], forward returns actual batches
- [ ] CIFAR-10

### Node definitions — Layers
- [ ] Conv2d — properties: outChannels, kernelSize, stride, padding. Shape executor: shape math.
- [ ] Linear — properties: outFeatures. Shape executor: shape math.
- [ ] BatchNorm
- [ ] MaxPool2d
- [ ] Dropout
- [ ] Flatten
- [ ] Reshape

### Node definitions — Activations
- [ ] ReLU, Sigmoid, Softmax (pass-through shapes)

### Node definitions — Structural
- [ ] Concat — dynamic ports (numInputs property)
- [ ] Split
- [ ] Residual/Skip connections

### Node definitions — Loss
- [ ] CrossEntropyLoss, MSELoss — take prediction + target, output scalar

### Node definitions — Optimizers
- [ ] SGD, Adam — not in data flow, read by training executor. Properties: lr, momentum, weight decay.

### Subgraph nodes
- [ ] GraphInput / GraphOutput sentinel nodes
- [ ] SubGraph node — getPorts reads inner graph's sentinels
- [ ] Pre-built compounds: Transformer Block, ResNet Block
- [ ] Export/import subgraphs as JSON

## Layer 6: UI

React + React Flow. Wires the visual editor to the engine built in layers 1–5.

### Canvas
- [ ] Replace hardcoded Phase 1 graph with engine-backed state
- [ ] Node rendering driven by registry — header, ports (colored by type), properties, metadata
- [ ] Edge rendering — wire color from type registry, shape annotations from upstream outputs
- [ ] Connection validation visual feedback (valid = snap, invalid = red / refuse)

### Node palette
- [ ] Sidebar populated from registry.list()
- [ ] Organized by category, searchable
- [ ] Drag from palette to canvas to add nodes

### Property inspector
- [ ] Select node → panel shows editable properties
- [ ] Auto-generated from getProperties() — number→input, boolean→toggle, select→dropdown
- [ ] Conditional visibility (e.g., padding mode only when padding > 0)
- [ ] Property change → triggers dirty tracking → shape mode re-runs → UI updates

### Real-time feedback
- [ ] Shape badges and param counts from lastResult.metadata
- [ ] Instant updates on property change (shape mode is eager)

### Subgraph navigation
- [ ] Double-click subgraph node to enter it
- [ ] Breadcrumb trail: Root > Transformer Block > Custom Attention
- [ ] Click breadcrumb to go back up

### Backend integration
- [ ] "Run" button → forward mode → serialize graph → send to FastAPI → stream activations back via WebSocket
- [ ] "Train" button → train mode → stream loss/metrics in real-time
- [ ] "Stop" button → cancel running execution
- [ ] Display activations, feature maps, attention maps, gradient heatmaps on nodes
- [ ] Live loss curve during training
- [ ] Weight heatmaps that update during training

### Polish
- [ ] Undo/redo
- [ ] Copy/paste nodes
- [ ] Share graphs as links
- [ ] Node search (Ctrl+K or similar)

## Python Backend

Separate from the layer system. Serves forward and train execution modes.

- [ ] FastAPI server
- [ ] Graph-to-PyTorch compiler — walk serialized nodes, construct torch.nn modules
- [ ] /forward endpoint — run forward pass, return activations per node
- [ ] /train endpoint — forward + backward + optimizer step loop
- [ ] WebSocket streaming for real-time results (training metrics, activations)
