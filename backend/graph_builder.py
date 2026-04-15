# Converts a serialized NodeTorch graph into PyTorch modules and executes them.
# Mirrors the TypeScript engine: topological sort → gather inputs → run module → store output.

import math
import torch
import torch.nn as nn

from node_builders import NODE_BUILDERS
from data_loaders import DATA_LOADERS, TRAIN_DATASETS, DENORMALIZERS


# Node types that take multiple named inputs instead of a single "in" port
LOSS_NODES = {"ml.loss.cross_entropy", "ml.loss.mse"}
OPTIMIZER_NODES = {"ml.optimizers.sgd", "ml.optimizers.adam", "ml.optimizers.adamw"}
# Structural nodes with multiple named inputs (passed as **kwargs)
MULTI_INPUT_NODES = {"ml.structural.add", "ml.structural.concat", "ml.layers.multihead_attention", "ml.layers.attention"}
SUBGRAPH_TYPE = "subgraph.block"
SENTINEL_INPUT = "subgraph.input"
SENTINEL_OUTPUT = "subgraph.output"

# Stores trained modules in memory so inference can reuse them.
# Key: "current" (single session for now), Value: dict of node_id → nn.Module
_model_store: dict[str, dict[str, nn.Module]] = {}

def has_trained_model() -> bool:
    return "current" in _model_store

def get_trained_modules() -> dict[str, nn.Module]:
    return _model_store.get("current", {})


def topological_sort(nodes: dict, edges: list) -> list[str]:
    """Kahn's algorithm — same logic as the TypeScript version."""
    downstream: dict[str, list[str]] = {nid: [] for nid in nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in nodes}

    for edge in edges:
        src = edge["source"]["nodeId"]
        tgt = edge["target"]["nodeId"]
        downstream[src].append(tgt)
        in_degree[tgt] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_ids = []

    while queue:
        nid = queue.pop(0)
        sorted_ids.append(nid)
        for child in downstream[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(sorted_ids) != len(nodes):
        raise ValueError("Graph contains a cycle")

    return sorted_ids


class SubGraphModule(nn.Module):
    """Wraps an inner graph's modules into a single nn.Module.
    Executes them in topological order, routing data through edges."""

    def __init__(self, inner_modules: dict[str, nn.Module], inner_nodes: dict,
                 inner_edges: list, inner_order: list[str]):
        super().__init__()
        self.inner_nodes = inner_nodes
        self.inner_edges = inner_edges
        self.inner_order = inner_order
        # Register inner modules so their parameters are visible
        self.inner_modules = nn.ModuleDict()
        for nid, mod in inner_modules.items():
            # Sanitize key for ModuleDict (no dots)
            safe_key = nid.replace('.', '_').replace('-', '_')
            self.inner_modules[safe_key] = mod
        # Keep a mapping from node_id → safe_key
        self._key_map = {nid: nid.replace('.', '_').replace('-', '_') for nid in inner_modules}

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        results: dict[str, dict[str, torch.Tensor]] = {}

        for node_id in self.inner_order:
            node = self.inner_nodes[node_id]
            node_type = node["type"]

            # Sentinel input: inject parent's inputs
            if node_type == SENTINEL_INPUT:
                results[node_id] = inputs
                continue

            # Sentinel output: collect and return
            if node_type == SENTINEL_OUTPUT:
                node_inputs = gather_inputs(node_id, self.inner_edges, results)
                results[node_id] = node_inputs
                continue

            # Skip optimizers/loss inside subgraph
            if node_type in OPTIMIZER_NODES or node_type in LOSS_NODES:
                continue

            node_inputs = gather_inputs(node_id, self.inner_edges, results)
            safe_key = self._key_map.get(node_id)
            if not safe_key or safe_key not in self.inner_modules:
                continue

            module = self.inner_modules[safe_key]

            if node_type in MULTI_INPUT_NODES:
                output = module(**{k: v for k, v in node_inputs.items()})
            elif "in" in node_inputs:
                output = module(node_inputs["in"])
            else:
                continue

            results[node_id] = {"out": output}

        # Find output sentinel's results
        for node_id in self.inner_order:
            if self.inner_nodes[node_id]["type"] == SENTINEL_OUTPUT:
                return results.get(node_id, {})

        return {}


def build_subgraph_module(subgraph_data: dict, parent_input_shapes: dict) -> SubGraphModule:
    """Build a SubGraphModule from a serialized subgraph."""
    nodes = {n["id"]: n for n in subgraph_data["nodes"]}
    edges = subgraph_data["edges"]
    order = topological_sort(nodes, edges)

    # First pass: determine shapes by doing a dry run with shape info
    inner_modules: dict[str, nn.Module] = {}
    shape_results: dict[str, dict[str, list]] = {}

    for node_id in order:
        node = nodes[node_id]
        node_type = node["type"]
        props = node.get("properties", {})

        if node_type == SENTINEL_INPUT:
            shape_results[node_id] = parent_input_shapes
            continue

        if node_type == SENTINEL_OUTPUT:
            continue

        if node_type in OPTIMIZER_NODES or node_type in LOSS_NODES:
            continue

        # Gather input shapes from upstream
        input_shapes: dict[str, list] = {}
        for edge in edges:
            if edge["target"]["nodeId"] == node_id:
                src_id = edge["source"]["nodeId"]
                src_port = edge["source"]["portId"]
                tgt_port = edge["target"]["portId"]
                if src_id in shape_results and src_port in shape_results[src_id]:
                    input_shapes[tgt_port] = shape_results[src_id][src_port]

        builder = NODE_BUILDERS.get(node_type)
        if not builder:
            continue

        module = builder(props, input_shapes)
        inner_modules[node_id] = module

        # Compute output shape for downstream
        if "in" in input_shapes:
            in_shape = input_shapes["in"]
            # Run a dummy tensor to get output shape
            dummy = torch.zeros(in_shape)
            with torch.no_grad():
                out = module(dummy)
            shape_results[node_id] = {"out": list(out.shape)}

    return SubGraphModule(inner_modules, nodes, edges, order)


def gather_inputs(
    node_id: str, edges: list, results: dict[str, dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """For a given node, collect input tensors from upstream nodes via edges."""
    inputs: dict[str, torch.Tensor] = {}
    for edge in edges:
        if edge["target"]["nodeId"] == node_id:
            src_id = edge["source"]["nodeId"]
            src_port = edge["source"]["portId"]
            tgt_port = edge["target"]["portId"]
            if src_id in results and src_port in results[src_id]:
                inputs[tgt_port] = results[src_id][src_port]
    return inputs


def build_and_run_graph(graph_data: dict) -> tuple[
    dict[str, nn.Module],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict],
    dict,
    list,
]:
    """
    Build modules and run a forward pass. Returns everything needed
    for both single forward and training.
    """
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    modules: dict[str, nn.Module] = {}
    results: dict[str, dict[str, torch.Tensor]] = {}
    node_results: dict[str, dict] = {}

    for node_id in order:
        node = nodes[node_id]
        node_type = node["type"]
        props = node["properties"]

        # --- Data nodes ---
        loader = DATA_LOADERS.get(node_type)
        if loader:
            try:
                tensors = loader(props)
                results[node_id] = tensors
                outputs = {k: tensor_info(v) for k, v in tensors.items()}
                first_tensor = next(iter(tensors.values()))
                node_results[node_id] = {
                    "outputs": outputs,
                    "metadata": {"outputShape": list(first_tensor.shape)},
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }
            continue

        # --- Optimizer nodes: skip during forward, handled by training ---
        if node_type in OPTIMIZER_NODES:
            node_results[node_id] = {
                "outputs": {},
                "metadata": {},
            }
            continue

        # --- Loss nodes: take named inputs (predictions + labels) ---
        if node_type in LOSS_NODES:
            inputs = gather_inputs(node_id, edges, results)
            if "predictions" not in inputs or "labels" not in inputs:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": "Connect both predictions and labels"},
                }
                continue

            builder = NODE_BUILDERS.get(node_type)
            if not builder:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": f"Unknown node type: {node_type}"},
                }
                continue

            input_shapes = {k: list(v.shape) for k, v in inputs.items()}
            try:
                module = builder(props, input_shapes)
                modules[node_id] = module
                loss = module(inputs["predictions"], inputs["labels"])
                results[node_id] = {"out": loss}
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(loss)},
                    "metadata": {
                        "outputShape": ["scalar"],
                        "lossValue": _safe_float(float(loss.detach())),
                    },
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }
            continue

        # --- Structural nodes: multiple named inputs ---
        if node_type in MULTI_INPUT_NODES:
            inputs = gather_inputs(node_id, edges, results)
            builder = NODE_BUILDERS.get(node_type)
            if not builder:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": f"Unknown node type: {node_type}"},
                }
                continue

            input_shapes = {k: list(v.shape) for k, v in inputs.items()}
            try:
                module = builder(props, input_shapes)
                modules[node_id] = module

                # Pass all inputs as keyword arguments
                output = module(**{k: v for k, v in inputs.items()})
                results[node_id] = {"out": output}
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(output)},
                    "metadata": {"outputShape": list(output.shape)},
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }
            continue

        # --- Subgraph nodes: build inner modules as a single nn.Module ---
        if node_type == SUBGRAPH_TYPE:
            subgraph_data = node.get("subgraph")
            if not subgraph_data:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": "Empty subgraph"},
                }
                continue

            inputs = gather_inputs(node_id, edges, results)
            input_shapes = {k: list(v.shape) for k, v in inputs.items()}
            try:
                sg_module = build_subgraph_module(subgraph_data, input_shapes)
                modules[node_id] = sg_module

                with torch.no_grad():
                    sg_outputs = sg_module(**inputs)

                # The subgraph returns a dict of port_id → tensor
                # Store the first output for downstream
                first_key = next(iter(sg_outputs), None)
                if first_key:
                    results[node_id] = {"out": sg_outputs[first_key]}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(sg_outputs[first_key])},
                        "metadata": {"outputShape": list(sg_outputs[first_key].shape)},
                    }
                else:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": "Subgraph produced no output"},
                    }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }
            continue

        # --- Layer nodes: single "in" input ---
        builder = NODE_BUILDERS.get(node_type)
        if not builder:
            node_results[node_id] = {
                "outputs": {},
                "metadata": {"error": f"Unknown node type: {node_type}"},
            }
            continue

        inputs = gather_inputs(node_id, edges, results)
        if "in" not in inputs:
            node_results[node_id] = {
                "outputs": {},
                "metadata": {"error": "No input connected"},
            }
            continue

        input_shapes = {k: list(v.shape) for k, v in inputs.items()}
        try:
            module = builder(props, input_shapes)
            modules[node_id] = module
            output = module(inputs["in"])
            results[node_id] = {"out": output}
            node_results[node_id] = {
                "outputs": {"out": tensor_info(output)},
                "metadata": {
                    "outputShape": list(output.shape),
                    "paramCount": sum(p.numel() for p in module.parameters()),
                },
            }
        except Exception as e:
            node_results[node_id] = {
                "outputs": {},
                "metadata": {"error": str(e)},
            }

    return modules, results, node_results, nodes, edges


def execute_graph(graph_data: dict) -> dict:
    """Run a single forward pass. Returns per-node results."""
    with torch.no_grad():
        _, _, node_results, _, _ = build_and_run_graph(graph_data)
    return node_results


def train_graph(graph_data: dict, on_epoch=None, cancel_event=None) -> dict:
    """
    Run a training loop. Finds the optimizer node, builds modules with
    gradient tracking, and trains for the specified number of epochs.

    cancel_event: optional threading.Event — if set, training stops after the current epoch.

    on_epoch: optional callback(epoch, loss, accuracy) for streaming progress.
    """
    graph = graph_data["graph"]
    nodes_list = graph["nodes"]

    # Find the optimizer node
    optimizer_node = None
    for n in nodes_list:
        if n["type"] in OPTIMIZER_NODES:
            optimizer_node = n
            break

    if not optimizer_node:
        return {"error": "No optimizer node in graph"}

    props = optimizer_node["properties"]
    epochs = props.get("epochs", 5)
    lr = props.get("lr", 0.01)
    momentum = props.get("momentum", 0.9)
    weight_decay = props.get("weightDecay", 0)

    # Find the data node to get batch size
    data_node = None
    for n in nodes_list:
        if DATA_LOADERS.get(n["type"]):
            data_node = n
            break

    if not data_node:
        return {"error": "No data node in graph"}

    batch_size = data_node["properties"].get("batchSize", 1)
    data_loader_fn = DATA_LOADERS[data_node["type"]]

    # Build modules once (first pass establishes shapes)
    modules, results, node_results, nodes, edges = build_and_run_graph(graph_data)

    # Collect all trainable parameters
    all_params = []
    for module in modules.values():
        all_params.extend(module.parameters())

    if not all_params:
        return {"error": "No trainable parameters in graph"}

    # Create optimizer
    opt_type = optimizer_node["type"]
    if opt_type == "ml.optimizers.adam":
        optimizer = torch.optim.Adam(
            all_params,
            lr=lr,
            betas=(props.get("beta1", 0.9), props.get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
    elif opt_type == "ml.optimizers.adamw":
        optimizer = torch.optim.AdamW(
            all_params,
            lr=lr,
            betas=(props.get("beta1", 0.9), props.get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            all_params, lr=lr, momentum=momentum, weight_decay=weight_decay,
        )

    # Find loss node
    loss_node_id = None
    for nid, n in nodes.items():
        if n["type"] in LOSS_NODES:
            loss_node_id = nid
            break

    if not loss_node_id:
        return {"error": "No loss node in graph"}

    # Load full dataset for training
    dataset_type = data_node["type"]
    dataset_builder = TRAIN_DATASETS.get(dataset_type)
    if not dataset_builder:
        return {"error": f"Training not supported for dataset: {dataset_type}"}
    train_dataset = dataset_builder()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )

    # Training loop
    import time
    order = topological_sort(nodes, edges)
    epoch_results = []
    total_batches = len(train_loader)

    for epoch in range(epochs):
        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            break

        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if cancel_event and cancel_event.is_set():
                break
            optimizer.zero_grad()

            # Forward pass through the graph
            batch_results: dict[str, dict[str, torch.Tensor]] = {}

            for node_id in order:
                node = nodes[node_id]
                node_type = node["type"]

                # Data node: use current batch
                if DATA_LOADERS.get(node_type):
                    batch_results[node_id] = {"out": images, "labels": labels}
                    continue

                # Optimizer: skip
                if node_type in OPTIMIZER_NODES:
                    continue

                # Loss node
                if node_type in LOSS_NODES:
                    inputs = gather_inputs(node_id, edges, batch_results)
                    if "predictions" in inputs and "labels" in inputs:
                        loss = modules[node_id](inputs["predictions"], inputs["labels"])
                        batch_results[node_id] = {"out": loss}
                    continue

                # Structural nodes (Add, etc.)
                if node_type in MULTI_INPUT_NODES:
                    inputs = gather_inputs(node_id, edges, batch_results)
                    output = modules[node_id](**{k: v for k, v in inputs.items()})
                    batch_results[node_id] = {"out": output}
                    continue

                # Subgraph nodes
                if node_type == SUBGRAPH_TYPE:
                    inputs = gather_inputs(node_id, edges, batch_results)
                    sg_outputs = modules[node_id](**inputs)
                    first_key = next(iter(sg_outputs), None)
                    if first_key:
                        batch_results[node_id] = {"out": sg_outputs[first_key]}
                    continue

                # Layer node
                inputs = gather_inputs(node_id, edges, batch_results)
                if "in" in inputs:
                    output = modules[node_id](inputs["in"])
                    batch_results[node_id] = {"out": output}

            # Backward pass
            loss_tensor = batch_results.get(loss_node_id, {}).get("out")
            if loss_tensor is not None:
                loss_tensor.backward()
                optimizer.step()
                total_loss += loss_tensor.item()

                # Compute accuracy from the node before loss (predictions)
                # Find which node feeds predictions to the loss node
                for edge in edges:
                    if (edge["target"]["nodeId"] == loss_node_id
                            and edge["target"]["portId"] == "predictions"):
                        pred_node_id = edge["source"]["nodeId"]
                        preds = batch_results.get(pred_node_id, {}).get("out")
                        if preds is not None:
                            predicted = preds.argmax(dim=1)
                            correct += (predicted == labels).sum().item()
                            total += labels.size(0)
                        break

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0

        epoch_result = {
            "epoch": epoch + 1,
            "totalEpochs": epochs,
            "loss": _safe_float(avg_loss),
            "accuracy": _safe_float(accuracy),
            "time": round(epoch_time, 2),
            "batches": total_batches,
            "samples": total,
        }
        epoch_results.append(epoch_result)

        if on_epoch:
            on_epoch(epoch_result)

    # Final forward pass to get updated node results
    with torch.no_grad():
        final_results: dict[str, dict[str, torch.Tensor]] = {}
        node_results = {}

        for node_id in order:
            node = nodes[node_id]
            node_type = node["type"]

            if DATA_LOADERS.get(node_type):
                tensors = data_loader_fn(node["properties"])
                final_results[node_id] = tensors
                outputs = {k: tensor_info(v) for k, v in tensors.items()}
                first_tensor = next(iter(tensors.values()))
                node_results[node_id] = {
                    "outputs": outputs,
                    "metadata": {"outputShape": list(first_tensor.shape)},
                }
                continue

            if node_type in OPTIMIZER_NODES:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {
                        "finalLoss": epoch_results[-1]["loss"] if epoch_results else None,
                        "finalAccuracy": epoch_results[-1]["accuracy"] if epoch_results else None,
                    },
                }
                continue

            if node_type in LOSS_NODES:
                inputs = gather_inputs(node_id, edges, final_results)
                if "predictions" in inputs and "labels" in inputs:
                    loss = modules[node_id](inputs["predictions"], inputs["labels"])
                    final_results[node_id] = {"out": loss}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(loss)},
                        "metadata": {
                            "outputShape": ["scalar"],
                            "lossValue": _safe_float(float(loss.detach())),
                        },
                    }
                continue

            inputs = gather_inputs(node_id, edges, final_results)

            if node_type == SUBGRAPH_TYPE:
                sg_outputs = modules[node_id](**inputs)
                first_key = next(iter(sg_outputs), None)
                if first_key:
                    final_results[node_id] = {"out": sg_outputs[first_key]}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(sg_outputs[first_key])},
                        "metadata": {
                            "outputShape": list(sg_outputs[first_key].shape),
                            "paramCount": sum(p.numel() for p in modules[node_id].parameters()),
                        },
                    }
            elif node_type in MULTI_INPUT_NODES:
                output = modules[node_id](**{k: v for k, v in inputs.items()})
                final_results[node_id] = {"out": output}
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(output)},
                    "metadata": {"outputShape": list(output.shape)},
                }
            elif "in" in inputs:
                output = modules[node_id](inputs["in"])
                final_results[node_id] = {"out": output}
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(output)},
                    "metadata": {
                        "outputShape": list(output.shape),
                        "paramCount": sum(p.numel() for p in modules[node_id].parameters()),
                    },
                }

    # Store trained modules for inference
    _model_store["current"] = modules

    return {
        "nodeResults": node_results,
        "epochs": epoch_results,
    }


def infer_graph(graph_data: dict) -> dict:
    """
    Run inference using trained weights.
    Loads a single sample, runs through stored trained modules,
    returns per-node results + prediction.
    """
    if not has_trained_model():
        return {"error": "No trained model — train first"}

    trained_modules = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    results: dict[str, dict[str, torch.Tensor]] = {}
    node_results: dict[str, dict] = {}
    prediction = None

    with torch.no_grad():
        for node_id in order:
            node = nodes[node_id]
            node_type = node["type"]
            props = node["properties"]

            # Data nodes: load a single sample
            loader = DATA_LOADERS.get(node_type)
            if loader:
                try:
                    # Override batch size to 1 for inference
                    infer_props = {**props, "batchSize": 1}
                    tensors = loader(infer_props)
                    results[node_id] = tensors
                    outputs = {k: tensor_info(v) for k, v in tensors.items()}
                    first_tensor = next(iter(tensors.values()))

                    # Include the actual label and image pixels for display
                    label = int(tensors["labels"][0]) if "labels" in tensors else None

                    # Send raw pixel data for image preview
                    image_data = None
                    image_channels = None
                    if "out" in tensors and tensors["out"].dim() == 4:
                        img = tensors["out"][0]  # [C, H, W]
                        C = img.shape[0]

                        denorm = DENORMALIZERS.get(node_type)
                        if denorm:
                            img = denorm(img)

                        img = (img.clamp(0, 1) * 255).byte()

                        if C == 1:
                            image_data = img[0].tolist()
                            image_channels = 1
                        else:
                            image_data = img.permute(1, 2, 0).tolist()
                            image_channels = C

                    node_results[node_id] = {
                        "outputs": outputs,
                        "metadata": {
                            "outputShape": list(first_tensor.shape),
                            "actualLabel": label,
                            "imagePixels": image_data,
                            "imageChannels": image_channels,
                        },
                    }
                except Exception as e:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": str(e)},
                    }
                continue

            # Optimizer/loss: skip during inference
            if node_type in OPTIMIZER_NODES or node_type in LOSS_NODES:
                node_results[node_id] = {"outputs": {}, "metadata": {}}
                continue

            # Layer / structural nodes: use trained modules
            module = trained_modules.get(node_id)
            if not module:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": "No trained module for this node"},
                }
                continue

            inputs = gather_inputs(node_id, edges, results)

            # Subgraph nodes
            if node_type == SUBGRAPH_TYPE:
                try:
                    sg_outputs = module(**inputs)
                    first_key = next(iter(sg_outputs), None)
                    if first_key:
                        results[node_id] = {"out": sg_outputs[first_key]}
                        node_results[node_id] = {
                            "outputs": {"out": tensor_info(sg_outputs[first_key])},
                            "metadata": {"outputShape": list(sg_outputs[first_key].shape)},
                        }
                except Exception as e:
                    node_results[node_id] = {"outputs": {}, "metadata": {"error": str(e)}}
                continue

            # Structural nodes pass all named inputs
            if node_type in MULTI_INPUT_NODES:
                try:
                    output = module(**{k: v for k, v in inputs.items()})
                    results[node_id] = {"out": output}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(output)},
                        "metadata": {"outputShape": list(output.shape)},
                    }
                except Exception as e:
                    node_results[node_id] = {"outputs": {}, "metadata": {"error": str(e)}}
                continue

            if "in" not in inputs:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": "No input connected"},
                }
                continue

            try:
                output = module(inputs["in"])
                results[node_id] = {"out": output}

                meta: dict = {
                    "outputShape": list(output.shape),
                    "paramCount": sum(p.numel() for p in module.parameters()),
                }

                # If this is the last layer before loss (output is [1, num_classes]),
                # compute prediction
                is_final = any(
                    e["source"]["nodeId"] == node_id
                    and nodes.get(e["target"]["nodeId"], {}).get("type") in LOSS_NODES
                    for e in edges
                )
                if is_final and output.dim() == 2:
                    probs = torch.softmax(output, dim=1)[0]
                    predicted_class = int(probs.argmax())
                    confidence = float(probs[predicted_class])
                    prediction = {
                        "predictedClass": predicted_class,
                        "confidence": _safe_float(confidence),
                        "probabilities": [_safe_float(float(p)) for p in probs],
                    }
                    meta["prediction"] = prediction

                node_results[node_id] = {
                    "outputs": {"out": tensor_info(output)},
                    "metadata": meta,
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }

    return {
        "nodeResults": node_results,
        "prediction": prediction,
    }


def _safe_float(v: float) -> float | None:
    """Convert to JSON-safe float. NaN and Inf become None."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def tensor_info(t: torch.Tensor) -> dict:
    """Extract displayable info from a tensor (not the raw data)."""
    ft = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": _safe_float(float(ft.mean())),
        "std": _safe_float(float(ft.std())) if ft.numel() > 1 else 0.0,
        "min": _safe_float(float(ft.min())),
        "max": _safe_float(float(ft.max())),
    }
