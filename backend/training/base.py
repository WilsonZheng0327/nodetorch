"""Shared training infrastructure — context, utilities, result handling.

All training loop plugins use these building blocks. Adding a new paradigm
(GAN, diffusion, etc.) shouldn't require duplicating any of this.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import threading
import time
import inspect

import torch
import torch.nn as nn
import torch.utils.data

from graph_builder import (
    build_and_run_graph,
    topological_sort,
    gather_inputs,
    get_device,
    _safe_float,
    _model_store,
    _last_run,
    ALL_LOSS_NODES,
    OPTIMIZER_NODES,
    LOSS_NODES,
    SUBGRAPH_TYPE,
    SENTINEL_INPUT,
    SENTINEL_OUTPUT,
    SubGraphModule,
    module_weight_info,
    gradient_info,
    batchnorm_info,
    activation_info,
    tensor_info,
    _pick_tracked_samples,
    _probe_tracked_samples,
    _collect_misclassifications,
)
from data_loaders import DATA_LOADERS, TRAIN_DATASETS, CLASS_NAMES
from forward_utils import run_forward_pass


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TrainingContext:
    """Everything a training loop needs. Built once, passed to the loop function."""

    # Raw graph data (for run history serialization)
    graph_data: dict

    # Parsed graph
    nodes: dict                     # node_id → node dict
    edges: list                     # edge list
    order: list[str]                # topological order

    # Built modules
    modules: dict[str, nn.Module]
    initial_node_results: dict      # from build_and_run_graph (for final forward fallback)

    # Key nodes
    data_node: dict
    data_node_id: str
    loss_node_ids: list[str]        # list for GAN (multiple losses)
    optimizer_nodes: list[dict]     # list for GAN (multiple optimizers)
    dataset_type: str               # e.g. "data.mnist"

    # Hyperparameters (from primary optimizer)
    epochs: int
    batch_size: int
    grad_clip_norm: float
    early_stop_patience: int
    val_split: float

    # Data
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader | None
    train_dataset: torch.utils.data.Dataset
    data_loader_fn: callable        # for final forward pass

    # Tracked samples (picked before training starts)
    tracked_samples: list[dict]

    # Callbacks
    on_epoch: callable | None
    on_batch: callable | None
    cancel_event: threading.Event | None


@dataclass
class TrainingResult:
    """What every training loop returns."""

    epoch_results: list[dict] = field(default_factory=list)
    modules: dict[str, nn.Module] = field(default_factory=dict)
    node_results: dict = field(default_factory=dict)
    final_results: dict = field(default_factory=dict)
    confusion_data: dict | None = None
    misclassifications: list | None = None
    training_mode: str = "standard"
    error: str | None = None


# ============================================================================
# Context builder (setup phase)
# ============================================================================

def build_training_context(
    graph_data: dict,
    on_epoch=None,
    on_batch=None,
    cancel_event=None,
) -> TrainingContext | dict:
    """Build the shared training context from graph data.

    Returns TrainingContext on success, or {"error": "..."} dict on failure.
    """
    graph = graph_data["graph"]
    nodes_list = graph["nodes"]

    # Find optimizer node(s)
    optimizer_nodes = [n for n in nodes_list if n["type"] in OPTIMIZER_NODES]
    if not optimizer_nodes:
        return {"error": "No optimizer node in graph"}

    # Primary optimizer (first one — standard mode uses this)
    primary_opt = optimizer_nodes[0]
    props = primary_opt["properties"]
    epochs = props.get("epochs", 5)
    grad_clip_norm = float(props.get("gradClip", 0) or 0)
    early_stop_patience = int(props.get("earlyStopPatience", 0) or 0)
    val_split = props.get("valSplit", 0.1)

    # Seed
    seed = props.get("seed", 42)
    if seed is not None:
        import random as _random
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except ImportError:
            pass

    # Find data node
    data_node = None
    for n in nodes_list:
        if DATA_LOADERS.get(n["type"]):
            data_node = n
            break
    if not data_node:
        return {"error": "No data node in graph"}

    batch_size = data_node["properties"].get("batchSize", 1)
    data_loader_fn = DATA_LOADERS[data_node["type"]]

    # Build modules
    modules, results, node_results, nodes, edges = build_and_run_graph(graph_data)

    # Collect all trainable parameters
    all_params = list(p for m in modules.values() for p in m.parameters())
    if not all_params:
        return {"error": "No trainable parameters in graph"}

    # Find loss node(s)
    loss_node_ids = [nid for nid, n in nodes.items() if n["type"] in ALL_LOSS_NODES]
    if not loss_node_ids:
        return {"error": "No loss node in graph"}

    # Load dataset
    dataset_type = data_node["type"]
    train_dataset, train_loader, val_loader = load_dataset(
        dataset_type, data_node, props, batch_size,
    )
    if isinstance(train_dataset, dict):
        return train_dataset  # error dict

    # Pick tracked samples
    tracked_samples = _pick_tracked_samples(train_dataset, dataset_type, n=4)

    return TrainingContext(
        graph_data=graph_data,
        nodes=nodes,
        edges=edges,
        order=topological_sort(nodes, edges),
        modules=modules,
        initial_node_results=node_results,
        data_node=data_node,
        data_node_id=data_node["id"],
        loss_node_ids=loss_node_ids,
        optimizer_nodes=optimizer_nodes,
        dataset_type=dataset_type,
        epochs=epochs,
        batch_size=batch_size,
        grad_clip_norm=grad_clip_norm,
        early_stop_patience=early_stop_patience,
        val_split=val_split,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        data_loader_fn=data_loader_fn,
        tracked_samples=tracked_samples,
        on_epoch=on_epoch,
        on_batch=on_batch,
        cancel_event=cancel_event,
    )


# ============================================================================
# Shared utilities
# ============================================================================

def load_dataset(dataset_type, data_node, optimizer_props, batch_size):
    """Load dataset and create train/val DataLoaders.

    Returns (dataset, train_loader, val_loader) or an error dict.
    """
    dataset_builder = TRAIN_DATASETS.get(dataset_type)
    if not dataset_builder:
        return {"error": f"Training not supported for dataset: {dataset_type}"}

    data_props = data_node.get("properties", {})
    sig = inspect.signature(dataset_builder)
    if sig.parameters:
        train_dataset = dataset_builder(**{
            k: data_props[k] for k in sig.parameters if k in data_props
        })
    else:
        train_dataset = dataset_builder()

    val_split = optimizer_props.get("valSplit", 0.1)
    val_loader = None
    if val_split > 0 and len(train_dataset) > 10:
        n_val = max(1, int(len(train_dataset) * val_split))
        n_train = len(train_dataset) - n_val
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, train_loader, val_loader


def build_optimizer(optimizer_node: dict, params) -> torch.optim.Optimizer:
    """Create optimizer from node properties."""
    props = optimizer_node["properties"]
    opt_type = optimizer_node["type"]
    lr = props.get("lr", 0.01)
    weight_decay = props.get("weightDecay", 0)
    momentum = props.get("momentum", 0.9)

    if opt_type == "ml.optimizers.adam":
        return torch.optim.Adam(
            params, lr=lr,
            betas=(props.get("beta1", 0.9), props.get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
    elif opt_type == "ml.optimizers.adamw":
        return torch.optim.AdamW(
            params, lr=lr,
            betas=(props.get("beta1", 0.9), props.get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
    else:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def build_scheduler(optimizer, scheduler_type: str, epochs: int):
    """Create LR scheduler from type string."""
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "warmup":
        warmup_epochs = max(1, epochs // 10)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0,
        )
    return None


def init_weight_norms(modules: dict[str, nn.Module]) -> dict[str, float]:
    """Compute initial weight norms for delta tracking."""
    norms = {}
    for nid, mod in modules.items():
        params = list(mod.parameters())
        if params:
            norms[nid] = float(torch.cat([p.detach().flatten() for p in params]).float().norm())
    return norms


def compute_batch_accuracy(batch_results, loss_node_id, edges, labels):
    """Compute classification accuracy for a batch. Returns (correct, total, per_class_correct, per_class_total).

    Returns (0, 0, {}, {}) if not a classification task.
    """
    correct = 0
    total = 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}

    for edge in edges:
        if edge["target"]["nodeId"] == loss_node_id and edge["target"]["portId"] == "predictions":
            pred_node_id = edge["source"]["nodeId"]
            preds = batch_results.get(pred_node_id, {}).get("out")
            if preds is not None and preds.dim() == 2:
                predicted = preds.argmax(dim=1)
                correct = int((predicted == labels).sum().item())
                total = int(labels.size(0))
                for cls in labels.unique().tolist():
                    cls = int(cls)
                    mask = labels == cls
                    per_class_total[cls] = mask.sum().item()
                    per_class_correct[cls] = int((predicted[mask] == cls).sum().item())
            break

    return correct, total, per_class_correct, per_class_total


def run_validation_pass(ctx: TrainingContext, modules, loss_node_id):
    """Run validation and return (val_loss, val_accuracy)."""
    if ctx.val_loader is None:
        return None, None

    val_total_loss = 0.0
    val_correct = 0
    val_total = 0
    dev = get_device()

    with torch.no_grad():
        for images, labels in ctx.val_loader:
            if ctx.cancel_event and ctx.cancel_event.is_set():
                break
            images, labels = images.to(dev), labels.to(dev)
            data_inputs = {ctx.data_node_id: {"out": images, "labels": labels}}

            batch_results = run_forward_pass(modules, ctx.nodes, ctx.edges, ctx.order, data_inputs)

            v_loss = batch_results.get(loss_node_id, {}).get("out")
            if v_loss is not None:
                val_total_loss += float(v_loss.item())
                c, t, _, _ = compute_batch_accuracy(batch_results, loss_node_id, ctx.edges, labels)
                val_correct += c
                val_total += t

    n_batches = len(ctx.val_loader)
    val_loss = val_total_loss / n_batches if n_batches > 0 else None
    val_accuracy = val_correct / val_total if val_total > 0 else None
    return val_loss, val_accuracy


def collect_node_snapshots(modules, last_batch_results, nodes, order, prev_weight_norms):
    """Collect per-node visualization snapshots (weights, gradients, activations)."""
    node_snapshots = {}
    for node_id, module in modules.items():
        if isinstance(module, SubGraphModule):
            inner_snaps = {}
            inner_results = getattr(module, '_last_results', {})
            for inner_nid, inner_node in module.inner_nodes.items():
                inner_type = inner_node["type"]
                if inner_type == SENTINEL_INPUT:
                    t = inner_results.get(inner_nid, {}).get("in")
                    if t is not None and isinstance(t, torch.Tensor):
                        inner_snaps[inner_nid] = {"activations": activation_info(t)}
                elif inner_type == SENTINEL_OUTPUT:
                    t = inner_results.get(inner_nid, {}).get("out")
                    if t is not None and isinstance(t, torch.Tensor):
                        inner_snaps[inner_nid] = {"activations": activation_info(t)}
            for inner_nid, safe_key in module._key_map.items():
                if safe_key not in module.inner_modules:
                    continue
                inner_mod = module.inner_modules[safe_key]
                s = {}
                wi = module_weight_info(inner_mod)
                if wi:
                    s["weights"] = wi
                gi = gradient_info(inner_mod)
                if gi:
                    s["gradients"] = gi
                inner_out = inner_results.get(inner_nid, {}).get("out")
                if inner_out is not None and isinstance(inner_out, torch.Tensor):
                    s["activations"] = activation_info(inner_out)
                if s:
                    inner_snaps[inner_nid] = s
            block_snap: dict = {"innerSnapshots": inner_snaps} if inner_snaps else {}
            out_tensor = last_batch_results.get(node_id, {}).get("out")
            if out_tensor is not None and isinstance(out_tensor, torch.Tensor):
                block_snap["activations"] = activation_info(out_tensor)
            if block_snap:
                node_snapshots[node_id] = block_snap
            continue

        snap = {}
        wi = module_weight_info(module)
        if wi:
            snap["weights"] = wi
        gi = gradient_info(module)
        if gi:
            snap["gradients"] = gi
        bi = batchnorm_info(module)
        if bi:
            snap["batchnorm"] = bi
        params = list(module.parameters())
        if params and node_id in prev_weight_norms:
            cur_norm = float(torch.cat([p.detach().flatten() for p in params]).float().norm())
            snap["weightDelta"] = _safe_float(abs(cur_norm - prev_weight_norms[node_id]))
            prev_weight_norms[node_id] = cur_norm
        out_tensor = last_batch_results.get(node_id, {}).get("out")
        if out_tensor is not None and isinstance(out_tensor, torch.Tensor):
            snap["activations"] = activation_info(out_tensor)
        if snap:
            node_snapshots[node_id] = snap

    return node_snapshots


def build_gradient_flow(node_snapshots, nodes, order):
    """Build gradient norm summary per trainable layer."""
    grad_entries = []
    for nid in order:
        snap = node_snapshots.get(nid)
        if not snap:
            continue
        if "gradients" in snap:
            short = nodes[nid]["type"].split(".")[-1]
            grad_entries.append((short, snap["gradients"]["norm"]))
        inner = snap.get("innerSnapshots")
        if inner:
            block_prefix = nodes[nid].get("properties", {}).get("blockName") or nodes[nid]["type"].split(".")[-1]
            for inner_nid, inner_snap in inner.items():
                if "gradients" in inner_snap:
                    grad_entries.append((f"{block_prefix}/{inner_nid}", inner_snap["gradients"]["norm"]))

    name_counts: dict[str, int] = {}
    for short, _ in grad_entries:
        name_counts[short] = name_counts.get(short, 0) + 1
    name_idx: dict[str, int] = {}
    gradient_flow = []
    for short, norm in grad_entries:
        if name_counts[short] > 1:
            name_idx[short] = name_idx.get(short, 0) + 1
            label = f"{short}_{name_idx[short]}"
        else:
            label = short
        gradient_flow.append({"name": label, "norm": norm})
    return gradient_flow


def build_epoch_result(epoch, ctx, avg_loss, accuracy, val_loss, val_accuracy,
                       current_lr, epoch_time, total_batches, total,
                       gradient_flow, per_class_accuracy, node_snapshots, tracked_probes):
    """Assemble the epoch result dict for streaming to frontend."""
    return {
        "epoch": epoch + 1,
        "totalEpochs": ctx.epochs,
        "loss": _safe_float(avg_loss),
        "accuracy": _safe_float(accuracy),
        "valLoss": _safe_float(val_loss) if val_loss is not None else None,
        "valAccuracy": _safe_float(val_accuracy) if val_accuracy is not None else None,
        "learningRate": _safe_float(current_lr),
        "time": round(epoch_time, 2),
        "batches": total_batches,
        "samples": total,
        "gradientFlow": gradient_flow,
        "perClassAccuracy": per_class_accuracy,
        "nodeSnapshots": node_snapshots,
        "trackedSamples": tracked_probes,
    }


def run_final_forward(ctx, modules):
    """Post-training forward pass to get display metadata for each node."""
    from graph_builder import MULTI_INPUT_NODES, GAN_NOISE_TYPE

    final_results: dict = {}
    node_results: dict = {}

    with torch.no_grad():
        for node_id in ctx.order:
            node = ctx.nodes[node_id]
            node_type = node["type"]

            if DATA_LOADERS.get(node_type):
                tensors = ctx.data_loader_fn(node["properties"])
                final_results[node_id] = tensors
                outputs = {k: tensor_info(v) for k, v in tensors.items() if isinstance(v, torch.Tensor)}
                first_tensor = next(v for v in tensors.values() if isinstance(v, torch.Tensor))
                node_results[node_id] = {
                    "outputs": outputs,
                    "metadata": {"outputShape": list(first_tensor.shape)},
                }
                continue

            # GAN noise input: produce dummy noise
            if node_type == GAN_NOISE_TYPE:
                props = node.get("properties", {})
                batch_size = props.get("batchSize", 64)
                latent_dim = props.get("latentDim", 100)
                dev = get_device()
                dummy_noise = torch.randn(batch_size, latent_dim, device=dev)
                final_results[node_id] = {"out": dummy_noise}
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(dummy_noise)},
                    "metadata": {"outputShape": [batch_size, latent_dim]},
                }
                continue

            if node_type in OPTIMIZER_NODES:
                node_results[node_id] = {"outputs": {}, "metadata": {}}
                continue

            mod = modules.get(node_id)
            if mod is None:
                continue

            inputs = gather_inputs(node_id, ctx.edges, final_results)

            if node_type in LOSS_NODES:
                if "predictions" in inputs and "labels" in inputs:
                    loss = mod(inputs["predictions"], inputs["labels"])
                    final_results[node_id] = {"out": loss}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(loss)},
                        "metadata": {"outputShape": ["scalar"], "lossValue": _safe_float(float(loss.detach()))},
                    }
                continue

            if node_type in MULTI_INPUT_NODES:
                try:
                    output = mod(**{k: v for k, v in inputs.items()})
                    final_results[node_id] = {"out": output}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(output)},
                        "metadata": {"outputShape": list(output.shape) if hasattr(output, 'shape') else ["scalar"]},
                    }
                except Exception:
                    pass
                continue

            if node_type == SUBGRAPH_TYPE:
                try:
                    sg_outputs = mod(**inputs)
                    first_key = next(iter(sg_outputs), None)
                    if first_key:
                        final_results[node_id] = {"out": sg_outputs[first_key]}
                        node_results[node_id] = {
                            "outputs": {"out": tensor_info(sg_outputs[first_key])},
                            "metadata": {"outputShape": list(sg_outputs[first_key].shape)},
                        }
                except Exception:
                    pass
                continue

            if "in" in inputs:
                try:
                    raw = mod(inputs["in"])
                    output = raw if not isinstance(raw, dict) else next(iter(raw.values()))
                    if isinstance(raw, dict):
                        final_results[node_id] = raw
                    else:
                        final_results[node_id] = {"out": raw}
                    meta: dict = {
                        "outputShape": list(output.shape),
                        "paramCount": sum(p.numel() for p in mod.parameters()),
                    }
                    ai = activation_info(output) if isinstance(output, torch.Tensor) else None
                    if ai:
                        meta["activations"] = ai
                    out_info = {k: tensor_info(v) for k, v in raw.items()} if isinstance(raw, dict) else {"out": tensor_info(output)}
                    node_results[node_id] = {"outputs": out_info, "metadata": meta}
                except Exception:
                    pass

    return final_results, node_results


def save_training_results(ctx: TrainingContext, result: TrainingResult) -> dict:
    """Store training results in global state and return the response dict."""
    _model_store["current"] = result.modules

    _last_run.clear()
    _last_run["modules"] = result.modules
    _last_run["results"] = result.final_results
    _last_run["nodes"] = ctx.nodes
    _last_run["edges"] = ctx.edges

    if result.confusion_data:
        _last_run["confusionMatrix"] = result.confusion_data
    if result.misclassifications:
        _last_run["misclassifications"] = result.misclassifications

    # Save run to disk
    try:
        from runs_store import build_run_record, save_run
        record = build_run_record(
            graph_data=ctx.graph_data,
            epoch_results=result.epoch_results,
            training_time=sum(e.get("time", 0) for e in result.epoch_results),
            data_node=ctx.data_node,
            optimizer_node=ctx.optimizer_nodes[0],
        )
        save_run(record)
    except Exception:
        pass

    # Set optimizer metadata
    for nid, n in ctx.nodes.items():
        if n["type"] in OPTIMIZER_NODES:
            result.node_results[nid] = {
                "outputs": {},
                "metadata": {
                    "finalLoss": result.epoch_results[-1]["loss"] if result.epoch_results else None,
                    "finalAccuracy": result.epoch_results[-1]["accuracy"] if result.epoch_results else None,
                },
            }

    return {"nodeResults": result.node_results}
