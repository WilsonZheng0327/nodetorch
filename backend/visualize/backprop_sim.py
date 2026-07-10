"""Simulate one forward+backward step and return per-node gradient magnitudes.

Used to animate the flow of gradients backward through the graph — educational
visualization of backpropagation.

Runs a single mini-batch, does forward, computes loss, backward, and reads each
node's accumulated gradient norm. Top-level only — SubGraphModule returns the
aggregate norm across all inner params.

Also provides run_backward_step_through() which returns rich per-node backward
stages with gradient visualizations, stats, and educational insights — the
backward counterpart of step_through.run_step_through().
"""

from __future__ import annotations
import copy
import functools
import torch


def _no_cudnn(fn):
    """Run a function with cuDNN disabled. cuDNN's RNN (LSTM/GRU) backward can only
    run in training mode, but we build+backward in EVAL mode (for deterministic,
    dropout-free gradients). The native RNN kernels support eval-mode backward, so
    we drop to them for the whole build+backward+sweep. No-op on CPU."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cudnn.enabled = prev

    return wrapper

from engine.graph_builder import (
    build_and_run_graph,
    gather_inputs,
    topological_sort,
    get_device,
    has_trained_model,
    get_trained_modules,
    _safe_float,
    prediction_from_logits,
    OPTIMIZER_NODES,
    LOSS_NODES,
    ALL_LOSS_NODES,
    SUBGRAPH_TYPE,
    SENTINEL_INPUT,
    SENTINEL_OUTPUT,
    DIFFUSION_SCHEDULER_TYPE,
    SubGraphModule,
)
from dataprep.data_loaders import DATA_LOADERS, DENORMALIZERS, CLASS_NAMES
from engine.forward_utils import execute_node
from visualize.node_viz import get_backward_viz, compact_stats_with_norm, param_grad_stats


def simulate_backprop(graph_data: dict, batch_size: int = 4) -> dict:
    """Run a single forward+backward pass and return per-node gradient magnitudes
    in topological order. Frontend reverses this for the animation."""
    # Build modules (fresh or trained — doesn't matter much for visualization)
    modules, _, _, nodes, edges = build_and_run_graph(graph_data)

    data_nid = None
    loss_nid = None
    for nid, n in nodes.items():
        if n["type"] in DATA_LOADERS:
            data_nid = nid
        if n["type"] in ALL_LOSS_NODES:
            loss_nid = nid
    if not data_nid or not loss_nid:
        return {"error": "Graph must have a data node and a loss node"}

    order = topological_sort(nodes, edges)

    # Load a small batch
    loader = DATA_LOADERS[nodes[data_nid]["type"]]
    try:
        props = {**nodes[data_nid].get("properties", {}), "batchSize": batch_size}
        tensors = loader(props)
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    dev = get_device()
    images = tensors["out"].to(dev)
    labels = tensors.get("labels")
    if isinstance(labels, torch.Tensor):
        labels = labels.to(dev)

    # Forward pass (NOT under no_grad — we need gradients)
    for m in modules.values():
        m.train()
        m.zero_grad(set_to_none=False)

    batch_results: dict[str, dict] = {data_nid: {"out": images, "labels": labels}}
    for nid in order:
        if nid == data_nid:
            continue
        n = nodes[nid]
        ntype = n["type"]
        if ntype in OPTIMIZER_NODES:
            continue
        mod = modules.get(nid)
        if mod is None:
            continue
        inputs = gather_inputs(nid, edges, batch_results)
        try:
            out = execute_node(ntype, mod, inputs)
        except Exception as e:
            if ntype in ALL_LOSS_NODES:
                return {"error": f"Loss computation failed: {e}"}
            raise
        if out is not None:
            batch_results[nid] = out

    # Backward
    loss_tensor = batch_results.get(loss_nid, {}).get("out")
    if loss_tensor is None:
        return {"error": "Loss did not produce a value — check predictions/labels connections"}

    try:
        loss_tensor.backward()
    except Exception as e:
        return {"error": f"Backward failed: {e}"}

    # Collect per-top-level-node gradient norms in topological order
    flow = []
    for nid in order:
        n = nodes[nid]
        ntype = n["type"]
        if ntype in DATA_LOADERS or ntype in OPTIMIZER_NODES:
            continue
        mod = modules.get(nid)
        if mod is None:
            continue
        grads = [p.grad.detach().flatten() for p in mod.parameters() if p.grad is not None]
        if grads:
            norm = float(torch.cat(grads).float().norm())
            flow.append({"nodeId": nid, "norm": norm})
        else:
            # Still include non-trainable nodes so animation walks through them
            flow.append({"nodeId": nid, "norm": 0.0})

    return {"flow": flow, "loss": float(loss_tensor.detach().item())}


# --- Rich backward step-through ---

def _friendly_name(node_type: str) -> str:
    """Convert 'ml.layers.conv2d' → 'Conv2d'."""
    last = node_type.split(".")[-1]
    if "_" in last:
        parts = last.split("_")
        return "".join(p.capitalize() for p in parts)
    return last[0].upper() + last[1:]


def _tensor_to_preview_image(img: torch.Tensor, dataset_type: str) -> dict:
    """Convert [C, H, W] to displayable pixels."""
    denorm = DENORMALIZERS.get(dataset_type)
    if denorm:
        img = denorm(img.cpu())
    img = (img.clamp(0, 1) * 255).byte()
    C = img.shape[0]
    if C == 1:
        return {"imagePixels": img[0].tolist(), "imageChannels": 1}
    return {"imagePixels": img.permute(1, 2, 0).tolist(), "imageChannels": C}


def _retain_grad_recursive(modules: dict) -> None:
    """Call retain_grad() on all intermediate tensors inside SubGraphModules.

    Without this, PyTorch discards gradients for non-leaf tensors during backward,
    making it impossible to inspect per-layer gradients inside subgraph blocks.
    """
    for mod in modules.values():
        if not isinstance(mod, SubGraphModule):
            continue
        inner_results = getattr(mod, '_last_results', {})
        for out_dict in inner_results.values():
            if not isinstance(out_dict, dict):
                continue
            for v in out_dict.values():
                if isinstance(v, torch.Tensor) and v.requires_grad and not v.is_leaf:
                    v.retain_grad()
        # Recurse into nested subgraphs
        if hasattr(mod, 'inner_modules'):
            _retain_grad_recursive(dict(mod.inner_modules))


def _loss_for_sample(graph_data: dict, sample_idx: int) -> float | None:
    """Forward-only pass on one sample with trained weights; return its loss."""
    try:
        _, nodes, _, results = _build_with_trained(graph_data, sample_idx=sample_idx, train_mode=False)
    except Exception:
        return None
    loss_nid = next((nid for nid, n in nodes.items() if n["type"] in ALL_LOSS_NODES), None)
    if loss_nid is None:
        return None
    loss_tensor = results.get(loss_nid, {}).get("out")
    if loss_tensor is None or not isinstance(loss_tensor, torch.Tensor):
        return None
    try:
        return float(loss_tensor.detach().item())
    except Exception:
        return None


def _pick_hard_sample(graph_data: dict, k: int = 40, hard_enough: float = 1.0) -> int | None:
    """Scan up to k random candidates and return the highest-loss sample index.

    A well-trained classifier gets most random samples right (near-0 loss), so its
    backward view would be all-zeros — useless for learning. Scanning a batch of
    candidates finds a sample the model actually struggles on, where the gradients
    are meaningful. Stops early once a clearly-hard sample (loss > hard_enough) is
    found so the common case stays fast.

    Falls back to a single random sample (None) if the dataset can't be sized or
    nothing produced a loss — the caller then behaves exactly as before.
    """
    import random as _random
    from dataprep.data_loaders import TRAIN_DATASETS

    data_type = next(
        (n["type"] for n in graph_data["graph"]["nodes"] if n["type"] in DATA_LOADERS),
        None,
    )
    dataset_fn = TRAIN_DATASETS.get(data_type) if data_type else None
    if dataset_fn is None:
        return None
    try:
        n_samples = len(dataset_fn())
    except Exception:
        return None
    if n_samples <= 0:
        return None

    best_idx, best_loss = None, float("-inf")
    for _ in range(k):
        idx = _random.randint(0, n_samples - 1)
        loss = _loss_for_sample(graph_data, idx)
        if loss is not None and loss > best_loss:
            best_idx, best_loss = idx, loss
        if best_loss > hard_enough:  # clearly a sample the model gets wrong — good enough
            break
    return best_idx


def _input_grad(node_id: str, edges: list, results: dict):
    """∂L/∂input for a node — the gradient it passes back to the previous layer.

    Read from the (retained) grad of the node's input tensor, which is the upstream
    layer's output. So this layer's grad-out is literally the previous layer's
    grad-in. Returns the primary float input's gradient, or None.
    """
    try:
        inputs = gather_inputs(node_id, edges, results)
    except Exception:
        return None
    for t in inputs.values():
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.grad is not None:
            return t.grad.detach()
    return None


def _loss_seed(logits, true_label: int | None, loss_val, class_names: list | None) -> dict | None:
    """The seed of backprop at the loss node: prediction vs truth, the loss value,
    and the error signal ∂L/∂logits it emits (for softmax + cross-entropy this is
    exactly predicted − target). This is where every gradient is born — far more
    useful than the loss node's own grad-in, which is a trivial ∂L/∂L = 1.
    """
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        return None
    probs = torch.softmax(logits.detach(), dim=1)[0]
    n = probs.numel()
    # Emitted gradient: prefer the real grad, fall back to p − onehot.
    if logits.grad is not None:
        err = logits.grad.detach()[0]
    else:
        err = probs.clone()
        if true_label is not None and 0 <= true_label < n:
            err[true_label] = err[true_label] - 1.0
    k = min(6, n)
    top = torch.topk(probs, k)
    return {
        "logits": [_safe_float(float(x)) for x in logits.detach()[0].tolist()],
        "probabilities": [_safe_float(float(p)) for p in probs.tolist()],
        "errorSignal": [_safe_float(float(e)) for e in err.tolist()],
        "trueLabel": true_label,
        "trueLabelProb": (
            _safe_float(float(probs[true_label]))
            if true_label is not None and 0 <= true_label < n
            else None
        ),
        "loss": loss_val,
        "classNames": class_names,
        "topK": [
            {"index": int(i), "value": _safe_float(float(v))}
            for v, i in zip(top.values.tolist(), top.indices.tolist())
        ],
    }


def _linear_mechanism(module, out_grad, in_grad) -> dict | None:
    """The concrete chain-rule mechanism for a Linear layer: how the error at the
    output (∂L/∂output) becomes the error handed to the previous layer (∂L/∂input),
    routed back through the same weights — ∂L/∂input = ∂L/∂output · Wᵀ, i.e.
    in_grad[j] = Σ_i out_grad[i] · W[i, j].

    Returns the two error vectors plus a full trace of the most-blamed input neuron
    (its weighted sum of output errors), so the frontend can show the actual math.
    """
    if not isinstance(module, torch.nn.Linear) or out_grad is None or in_grad is None:
        return None
    W = module.weight.detach().float()  # [out, in]
    og = out_grad.detach().flatten().float()  # [out]
    ig = in_grad.detach().flatten().float()  # [in]
    if og.numel() != W.shape[0] or ig.numel() != W.shape[1]:
        return None  # not a plain [out]×[in] linear map (e.g. flattened batch) — skip

    out_dim, in_dim = int(W.shape[0]), int(W.shape[1])
    OUT_CAP, IN_CAP = 16, 32

    # Trace the most-blamed input neuron: its error = Σ_i out_grad[i] · W[i, j].
    j = int(ig.abs().argmax())
    col = W[:, j]  # [out] — the weights connecting every output back to input j
    products = og * col  # [out] — per-output contributions; they sum to ig[j]
    k = min(6, out_dim)
    top_idx = torch.topk(products.abs(), k).indices.tolist()

    return {
        "kind": "linear",
        "outDim": out_dim,
        "inDim": in_dim,
        "outGrad": [_safe_float(float(x)) for x in og[:OUT_CAP].tolist()],
        "inGrad": [_safe_float(float(x)) for x in ig[:IN_CAP].tolist()],
        "trace": {
            "inputIndex": j,
            "inputError": _safe_float(float(ig[j])),
            "shownTerms": k,
            "totalTerms": out_dim,
            "terms": [
                {
                    "out": int(i),
                    "outGrad": _safe_float(float(og[i])),
                    "weight": _safe_float(float(col[i])),
                    "product": _safe_float(float(products[i])),
                }
                for i in top_idx
            ],
        },
    }


def _sample_of_class(graph_data: dict, filter_label: int) -> int | None:
    """Resolve a random sample index belonging to the given class."""
    from dataprep.data_loaders import load_sample_by_label

    for n in graph_data["graph"]["nodes"]:
        if n["type"] in DATA_LOADERS:
            props = {**n.get("properties", {}), "batchSize": 1}
            try:
                _, idx = load_sample_by_label(n["type"], props, filter_label=filter_label)
                return idx if idx is not None and idx >= 0 else None
            except Exception:
                return None
    return None


@_no_cudnn
def run_backward_step_through(
    graph_data: dict, sample_idx: int | None = None, filter_label: int | None = None
) -> dict:
    """Run a forward+backward pass and return rich per-node backward stages.

    Requires trained weights. Accepts sample_idx to reproduce the same sample used
    in forward step-through, or filter_label to draw a sample of a specific class.

    Returns: { stages: [...], loss: float, sample: {...} }
    """
    if not has_trained_model():
        raise RuntimeError("Train the model first — backward step-through requires trained weights")

    graph_data = copy.deepcopy(graph_data)

    for n in graph_data["graph"]["nodes"]:
        if n["type"] in DATA_LOADERS:
            n["properties"] = {**n.get("properties", {}), "batchSize": 1}

    # A well-trained model classifies most random samples with ~0 loss, so the
    # whole backward view would show ~0 gradients everywhere — useless for
    # learning. When the caller doesn't pin a sample, pick the hardest of a few
    # random candidates (highest loss) so the gradients are actually meaningful.
    if sample_idx is None:
        sample_idx = (
            _sample_of_class(graph_data, filter_label)
            if filter_label is not None
            else _pick_hard_sample(graph_data)
        )

    # Eval mode: deterministic forward (no dropout noise) so the gradients are
    # reproducible and the loss shown here matches /one-step's "before" loss.
    try:
        modules, nodes, edges, results = _build_with_trained(graph_data, sample_idx=sample_idx, train_mode=False)
    except Exception as e:
        raise RuntimeError(f"Backward step-through failed: {e}")

    # Retain gradients on all subgraph inner tensors before backward
    _retain_grad_recursive(modules)

    order = topological_sort(nodes, edges)

    # Find data, loss, and the logits node feeding the loss ("predictions" port)
    data_nid, loss_nid, pred_nid = _find_key_nodes(nodes, edges)
    if not data_nid or not loss_nid:
        return {"error": "Graph must have a data node and a loss node"}

    class_names = CLASS_NAMES.get(nodes[data_nid]["type"]) if data_nid else None

    # Extract sample info for the header
    sample_info = {}
    true_label = None
    data_tensors = results.get(data_nid, {})
    if data_tensors:
        out = data_tensors.get("out")
        labels = data_tensors.get("labels")
        if labels is not None and isinstance(labels, torch.Tensor) and labels.dim() == 1:
            true_label = int(labels[0])
        sample_info = {
            "datasetType": nodes[data_nid]["type"],
            "actualLabel": true_label,
            "classNames": class_names,
        }
        if out is not None and isinstance(out, torch.Tensor) and out.dim() == 4:
            sample_info.update(_tensor_to_preview_image(out[0].detach(), nodes[data_nid]["type"]))
        elif out is not None and isinstance(out, torch.Tensor) and out.dim() == 2:
            sample_info["tokenIds"] = out[0].tolist()[:64]
            if "_texts" in data_tensors:
                sample_info["sampleText"] = data_tensors["_texts"][0][:500]

    # Backward pass
    loss_tensor = results.get(loss_nid, {}).get("out") if loss_nid else None
    if loss_tensor is None:
        # GAN graphs can't run backward in step-through (loss needs dual-path execution)
        return {"stages": [], "loss": 0.0, "sample": sample_info}

    try:
        loss_tensor.backward()
    except Exception as e:
        return {"error": f"Backward failed: {e}"}

    loss_val = _safe_float(float(loss_tensor.detach().item()))

    # Collect per-node gradient captures from hooks
    grad_captures = getattr(run_backward_step_through, '_grad_captures', {})

    # Build backward stages in reverse topological order
    reversed_order = list(reversed(order))
    stages: list[dict] = []

    for node_id in reversed_order:
        node = nodes[node_id]
        node_type = node["type"]

        if node_type in OPTIMIZER_NODES:
            continue

        # Handle subgraph blocks — flatten inner nodes
        if node_type == SUBGRAPH_TYPE:
            sg_module = modules.get(node_id)
            if isinstance(sg_module, SubGraphModule):
                block_name = node.get("properties", {}).get("blockName") or node_id
                inner_stages = _build_backward_subgraph_stages(
                    sg_module=sg_module,
                    block_name=block_name,
                    depth=1,
                    parent_path=[node_id],
                )
                stages.extend(inner_stages)
                continue

        module = modules.get(node_id)

        # Get activation from forward pass
        out_dict = results.get(node_id, {})
        activation = out_dict.get("out") if isinstance(out_dict, dict) else None

        # Get gradient (from hook or from activation's .grad)
        gradient = None
        if activation is not None and isinstance(activation, torch.Tensor) and activation.grad is not None:
            gradient = activation.grad.detach()
        elif node_id in grad_captures:
            gradient = grad_captures[node_id]

        # Skip data nodes in backward (they don't receive meaningful gradients)
        if node_type in DATA_LOADERS:
            continue

        # Get backward viz from registry
        viz_result = get_backward_viz(node_type, module, activation, gradient)

        stage: dict = {
            "stageId": f"bwd/{node_id}",
            "path": [node_id],
            "nodeId": node_id,
            "nodeType": node_type,
            "displayName": _friendly_name(node_type),
            "depth": 0,
            "inputShape": list(activation.shape) if activation is not None and isinstance(activation, torch.Tensor) else None,
            "outputShape": list(gradient.shape) if gradient is not None else None,
            "gradientShape": list(gradient.shape) if gradient is not None else None,
        }

        # Gradient stats — grad IN (∂L/∂output), arrives from the downstream layer
        if gradient is not None:
            stage["stats"] = compact_stats_with_norm(gradient)

        # Grad OUT (∂L/∂input) — what this layer passes back to the previous one
        grad_out = _input_grad(node_id, edges, results)
        if grad_out is not None:
            stage["gradOut"] = {
                "stats": compact_stats_with_norm(grad_out),
                "shape": list(grad_out.shape),
            }

        # Loss node: attach the backprop seed (loss + emitted error signal p − y)
        if node_type in ALL_LOSS_NODES and pred_nid:
            seed = _loss_seed(results.get(pred_nid, {}).get("out"), true_label, loss_val, class_names)
            if seed:
                stage["lossSeed"] = seed

        # Param grad stats (gradient-to-weight ratio)
        pgs = param_grad_stats(module)
        if pgs:
            stage["paramGradStats"] = pgs

        # The concrete chain-rule mechanism (Linear): out-error → in-error via Wᵀ
        mech = _linear_mechanism(module, gradient, grad_out)
        if mech:
            stage["mechanism"] = mech

        # Viz, extras, insight from registry
        if viz_result.get("viz"):
            stage["viz"] = viz_result["viz"]
        if viz_result.get("extras"):
            stage["extras"] = viz_result["extras"]
        if viz_result.get("insight"):
            stage["insight"] = viz_result["insight"]

        stages.append(stage)

    return {
        "stages": stages,
        "loss": loss_val,
        "sample": sample_info,
        "sampleIdx": sample_idx,  # so /one-step can reproduce this exact sample
    }


def _forward_pass(modules: dict, nodes: dict, edges: list) -> dict:
    """Run a forward pass with gradient tracking (no torch.no_grad)."""
    order = topological_sort(nodes, edges)
    dev = get_device()

    # Set modules to train mode and clear gradients
    for m in modules.values():
        if isinstance(m, torch.nn.Module):
            m.train()
            m.zero_grad(set_to_none=False)

    results: dict[str, dict] = {}

    for nid in order:
        n = nodes[nid]
        ntype = n["type"]

        # Data node: load a batch
        loader = DATA_LOADERS.get(ntype)
        if loader:
            props = n.get("properties", {})
            tensors = loader(props)
            tensors = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
            # Enable grad on float outputs for gradient capture (skip integer tensors like token IDs)
            if "out" in tensors and isinstance(tensors["out"], torch.Tensor) and tensors["out"].is_floating_point():
                tensors["out"] = tensors["out"].detach().requires_grad_(True)
            results[nid] = tensors
            continue

        if ntype in OPTIMIZER_NODES:
            continue

        # Diffusion scheduler: produce 3-port output (noisy, noise, timestep)
        if ntype == DIFFUSION_SCHEDULER_TYPE:
            inputs = gather_inputs(nid, edges, results)
            mod = modules.get(nid)
            if mod is not None and "images" in inputs:
                images = inputs["images"]
                noisy = images.clone().requires_grad_(True)
                noise_dummy = torch.zeros_like(images).requires_grad_(True)
                t_channel = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=dev).requires_grad_(True)
                results[nid] = {"out": noisy, "noise": noise_dummy, "timestep": t_channel}
                for v in results[nid].values():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        v.retain_grad()
            continue

        mod = modules.get(nid)
        if mod is None:
            continue

        inputs = gather_inputs(nid, edges, results)
        try:
            out = execute_node(ntype, mod, inputs)
        except Exception:
            continue

        if out is not None:
            # Retain grad on intermediate tensors so we can read them after backward
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    v.retain_grad()
            results[nid] = out

    return results


def _build_with_trained(graph_data: dict, sample_idx: int | None = None, train_mode: bool = True) -> tuple:
    """Build modules using trained weights and run forward pass.

    train_mode=False runs the modules in eval() mode (dropout off, BN frozen) so a
    forward pass is deterministic — needed for a clean before/after comparison.
    Gradients still flow in eval mode, so backward works either way.
    """
    from dataprep.data_loaders import load_sample_by_label
    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)
    dev = get_device()

    modules = {}
    results: dict[str, dict] = {}

    for node_id in order:
        node = nodes[node_id]
        node_type = node["type"]
        props = node.get("properties", {})

        if node_type in DATA_LOADERS:
            tensors, _ = load_sample_by_label(node_type, props, sample_idx=sample_idx)
            tensors = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
            if "out" in tensors and isinstance(tensors["out"], torch.Tensor) and tensors["out"].is_floating_point():
                tensors["out"] = tensors["out"].detach().requires_grad_(True)
            results[node_id] = tensors
            continue

        if node_type in OPTIMIZER_NODES:
            continue

        # Diffusion scheduler: produce 3-port output
        if node_type == DIFFUSION_SCHEDULER_TYPE:
            module = trained.get(node_id)
            if module is not None:
                modules[node_id] = module
                inputs = gather_inputs(node_id, edges, results)
                if "images" in inputs:
                    images = inputs["images"]
                    noisy = images.clone().requires_grad_(True)
                    noise_dummy = torch.zeros_like(images).requires_grad_(True)
                    t_channel = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=dev).requires_grad_(True)
                    results[node_id] = {"out": noisy, "noise": noise_dummy, "timestep": t_channel}
                    for v in results[node_id].values():
                        if isinstance(v, torch.Tensor) and v.requires_grad:
                            v.retain_grad()
            continue

        module = trained.get(node_id)
        if module is None:
            raise RuntimeError(f"Trained model missing module for node {node_id}")

        module.train() if train_mode else module.eval()
        module.zero_grad(set_to_none=False)
        modules[node_id] = module

        inputs = gather_inputs(node_id, edges, results)
        try:
            out = execute_node(node_type, module, inputs)
        except Exception:
            # Some nodes may fail (e.g. GAN loss needs both real/fake scores)
            continue
        if out is not None:
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    v.retain_grad()
            results[node_id] = out

    return modules, nodes, edges, results


def _build_backward_subgraph_stages(
    *,
    sg_module: SubGraphModule,
    block_name: str,
    depth: int,
    parent_path: list[str],
) -> list[dict]:
    """Walk a SubGraphModule's inner nodes in reverse and emit backward stages."""
    inner_nodes = sg_module.inner_nodes
    inner_edges = sg_module.inner_edges
    inner_order = sg_module.inner_order
    inner_results = getattr(sg_module, '_last_results', {})

    stages: list[dict] = []
    for inner_nid in reversed(inner_order):
        inner_node = inner_nodes[inner_nid]
        inner_type = inner_node["type"]

        if inner_type in (SENTINEL_INPUT, SENTINEL_OUTPUT):
            continue
        if inner_type in OPTIMIZER_NODES:
            continue

        safe_key = sg_module._key_map.get(inner_nid)
        inner_mod = sg_module.inner_modules[safe_key] if safe_key and safe_key in sg_module.inner_modules else None

        # Nested subgraph
        if isinstance(inner_mod, SubGraphModule):
            nested_name = inner_node.get("properties", {}).get("blockName") or inner_nid
            nested_stages = _build_backward_subgraph_stages(
                sg_module=inner_mod,
                block_name=nested_name,
                depth=depth + 1,
                parent_path=parent_path + [inner_nid],
            )
            stages.extend(nested_stages)
            continue

        # Get activation and gradient
        out_dict = inner_results.get(inner_nid, {})
        activation = out_dict.get("out") if isinstance(out_dict, dict) else None
        gradient = None
        if activation is not None and isinstance(activation, torch.Tensor) and activation.grad is not None:
            gradient = activation.grad.detach()

        viz_result = get_backward_viz(inner_type, inner_mod, activation, gradient)

        stage: dict = {
            "stageId": "/".join(parent_path + [inner_nid]),
            "path": parent_path + [inner_nid],
            "nodeId": inner_nid,
            "nodeType": inner_type,
            "displayName": _friendly_name(inner_type),
            "blockName": block_name,
            "depth": depth,
            "inputShape": list(activation.shape) if activation is not None and isinstance(activation, torch.Tensor) else None,
            "outputShape": list(gradient.shape) if gradient is not None else None,
            "gradientShape": list(gradient.shape) if gradient is not None else None,
        }

        if gradient is not None:
            stage["stats"] = compact_stats_with_norm(gradient)

        pgs = param_grad_stats(inner_mod)
        if pgs:
            stage["paramGradStats"] = pgs

        if viz_result.get("viz"):
            stage["viz"] = viz_result["viz"]
        if viz_result.get("extras"):
            stage["extras"] = viz_result["extras"]
        if viz_result.get("insight"):
            stage["insight"] = viz_result["insight"]

        stages.append(stage)

    return stages


# --- One gradient step: before/after ---

def _find_key_nodes(nodes: dict, edges: list) -> tuple:
    """Return (data_nid, loss_nid, pred_nid) — pred_nid is the node whose output
    feeds the loss node's 'predictions' port (the logits)."""
    data_nid = next((nid for nid, n in nodes.items() if n["type"] in DATA_LOADERS), None)
    loss_nid = next((nid for nid, n in nodes.items() if n["type"] in ALL_LOSS_NODES), None)
    pred_nid = None
    if loss_nid is not None:
        for e in edges:
            if e["target"]["nodeId"] == loss_nid and e["target"]["portId"] == "predictions":
                pred_nid = e["source"]["nodeId"]
                break
    return data_nid, loss_nid, pred_nid


def _optimizer_lr(nodes: dict) -> tuple[float, str | None]:
    """Read the learning rate (and optimizer type) from the graph's optimizer node."""
    for n in nodes.values():
        if n["type"] in OPTIMIZER_NODES:
            lr = n.get("properties", {}).get("lr")
            if isinstance(lr, (int, float)):
                return float(lr), n["type"]
            default = 0.001 if n["type"] in ("ml.optimizers.adam", "ml.optimizers.adamw") else 0.01
            return default, n["type"]
    return 0.01, None


@_no_cudnn
def run_one_step(graph_data: dict, sample_idx: int | None = None) -> dict:
    """Apply ONE gradient-descent step (W ← W − lr·∂L/∂W) on a single sample and
    report the before/after effect: loss, prediction, and per-layer weight change.

    The shared trained weights are snapshotted and restored, so the stored model is
    left untouched. Runs in eval mode so the only thing that changes between the
    before and after forward passes is the weight update itself.
    """
    if not has_trained_model():
        raise RuntimeError("Train the model first — one-step requires trained weights")

    from dataprep.data_loaders import CLASS_NAMES

    graph_data = copy.deepcopy(graph_data)
    for n in graph_data["graph"]["nodes"]:
        if n["type"] in DATA_LOADERS:
            n["properties"] = {**n.get("properties", {}), "batchSize": 1}

    if sample_idx is None:
        sample_idx = _pick_hard_sample(graph_data)

    try:
        modules, nodes, edges, results = _build_with_trained(
            graph_data, sample_idx=sample_idx, train_mode=False
        )
    except Exception as e:
        raise RuntimeError(f"One-step failed: {e}")

    data_nid, loss_nid, pred_nid = _find_key_nodes(nodes, edges)
    loss_tensor = results.get(loss_nid, {}).get("out") if loss_nid else None
    if loss_tensor is None:
        return {"error": "No loss value — check the predictions/labels connections"}

    # True label + class names for the prediction summary
    true_label = None
    labels = results.get(data_nid, {}).get("labels") if data_nid else None
    if isinstance(labels, torch.Tensor) and labels.dim() == 1 and labels.numel() > 0:
        true_label = int(labels[0])
    class_names = CLASS_NAMES.get(nodes[data_nid]["type"]) if data_nid else None

    loss_before = _safe_float(float(loss_tensor.detach().item()))
    pred_before = prediction_from_logits(
        results.get(pred_nid, {}).get("out") if pred_nid else None, true_label, top_k=5
    )

    # Snapshot every learnable parameter (clone — these tensors are the live weights)
    saved_params = {
        nid: [p.detach().clone() for p in m.parameters()]
        for nid, m in modules.items()
        if isinstance(m, torch.nn.Module) and any(True for _ in m.parameters())
    }

    # Backward + one SGD-style step
    for m in modules.values():
        if isinstance(m, torch.nn.Module):
            m.zero_grad(set_to_none=False)
    try:
        loss_tensor.backward()
    except Exception as e:
        return {"error": f"Backward failed: {e}"}

    lr, opt_type = _optimizer_lr(nodes)
    with torch.no_grad():
        for m in modules.values():
            if isinstance(m, torch.nn.Module):
                for p in m.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)

    # Per-layer weight change (compare the snapshot to the stepped weights)
    layers = []
    for nid, before_list in saved_params.items():
        m = modules[nid]
        after_list = [p.detach() for p in m.parameters()]
        before = torch.cat([b.flatten() for b in before_list]).float()
        after = torch.cat([a.flatten() for a in after_list]).float()
        delta = (after - before).abs()
        layers.append({
            "nodeId": nid,
            "displayName": _friendly_name(nodes[nid]["type"]),
            "weightNormBefore": _safe_float(float(before.norm())),
            "weightNormAfter": _safe_float(float(after.norm())),
            "meanAbsDelta": _safe_float(float(delta.mean())),
            "maxAbsDelta": _safe_float(float(delta.max())),
        })

    # Re-forward with the stepped weights to measure the after-state
    try:
        _, nodes2, edges2, results_after = _build_with_trained(
            graph_data, sample_idx=sample_idx, train_mode=False
        )
        _, loss_nid2, pred_nid2 = _find_key_nodes(nodes2, edges2)
        loss_after_tensor = results_after.get(loss_nid2, {}).get("out") if loss_nid2 else None
        loss_after = _safe_float(float(loss_after_tensor.detach().item())) if loss_after_tensor is not None else None
        pred_after = prediction_from_logits(
            results_after.get(pred_nid2, {}).get("out") if pred_nid2 else None, true_label, top_k=5
        )
    finally:
        # Restore the original weights so the stored model is unchanged
        with torch.no_grad():
            for nid, before_list in saved_params.items():
                for p, b in zip(modules[nid].parameters(), before_list):
                    p.copy_(b)

    return {
        "lr": lr,
        "optimizerType": opt_type,
        "sampleIdx": sample_idx,
        "trueLabel": true_label,
        "classNames": class_names,
        "loss": {"before": loss_before, "after": loss_after},
        "prediction": {"before": pred_before, "after": pred_after},
        "layers": layers,
    }


# --- Wiggle a weight: loss as a function of one weight ---

def _eval_point(
    modules: dict, nodes: dict, edges: list, data_out: dict, data_nid, loss_nid,
    pred_nid=None, true_label: int | None = None,
) -> dict:
    """Forward the existing (possibly weight-modified) modules on the cached sample.
    Returns the loss and, for classification, the predicted class + the probability
    the model gives the TRUE class. Runs under no_grad — just a measurement."""
    order = topological_sort(nodes, edges)
    res: dict[str, dict] = {data_nid: data_out}
    with torch.no_grad():
        for nid in order:
            if nid == data_nid:
                continue
            ntype = nodes[nid]["type"]
            if ntype in OPTIMIZER_NODES:
                continue
            mod = modules.get(nid)
            if mod is None:
                continue
            inputs = gather_inputs(nid, edges, res)
            try:
                out = execute_node(ntype, mod, inputs)
            except Exception:
                continue
            if out is not None:
                res[nid] = out

    lt = res.get(loss_nid, {}).get("out")
    loss = _safe_float(float(lt.detach().item())) if isinstance(lt, torch.Tensor) else None

    p_true, pred = None, None
    if pred_nid:
        logits = res.get(pred_nid, {}).get("out")
        if isinstance(logits, torch.Tensor) and logits.dim() == 2:
            probs = torch.softmax(logits.detach(), dim=1)[0]
            pred = int(probs.argmax())
            if true_label is not None and 0 <= true_label < probs.numel():
                p_true = _safe_float(float(probs[true_label]))
    return {"loss": loss, "pTrue": p_true, "pred": pred}


@_no_cudnn
def run_weight_wiggle(
    graph_data: dict,
    node_id: str,
    sample_idx: int | None = None,
    weight_index: int | None = None,
    num_points: int = 41,
) -> dict:
    """Sweep ONE weight of a Linear layer and report the loss at each value — the
    loss-vs-weight curve. The gradient ∂L/∂w is the slope of this curve at the
    current value, and a gradient step (w − lr·∂L/∂w) walks downhill along it.

    All other weights are frozen; only the chosen weight varies. The stored model
    is left untouched (the weight is restored afterwards).
    """
    if not has_trained_model():
        raise RuntimeError("Train the model first — wiggle requires trained weights")

    graph_data = copy.deepcopy(graph_data)
    for n in graph_data["graph"]["nodes"]:
        if n["type"] in DATA_LOADERS:
            n["properties"] = {**n.get("properties", {}), "batchSize": 1}

    if sample_idx is None:
        sample_idx = _pick_hard_sample(graph_data)

    modules, nodes, edges, results = _build_with_trained(
        graph_data, sample_idx=sample_idx, train_mode=False
    )
    data_nid, loss_nid, pred_nid = _find_key_nodes(nodes, edges)
    if node_id not in modules or not isinstance(modules[node_id], torch.nn.Linear):
        return {"error": "Wiggle currently supports Linear layers only"}

    # True label + class names — so we can show the prediction change as the weight
    # slides (the model's confidence in the correct answer moves with the loss).
    labels = results.get(data_nid, {}).get("labels")
    true_label = int(labels[0]) if isinstance(labels, torch.Tensor) and labels.dim() == 1 else None
    class_names = CLASS_NAMES.get(nodes[data_nid]["type"]) if data_nid else None

    loss_tensor = results.get(loss_nid, {}).get("out") if loss_nid else None
    if loss_tensor is None:
        return {"error": "No loss value — check the predictions/labels connections"}
    loss0 = _safe_float(float(loss_tensor.detach().item()))

    # Gradients (to read ∂L/∂w and to auto-pick the most impactful weight)
    for m in modules.values():
        if isinstance(m, torch.nn.Module):
            m.zero_grad(set_to_none=False)
    loss_tensor.backward()

    module = modules[node_id]
    W = module.weight
    G = W.grad
    if G is None:
        return {"error": "This layer received no gradient on this sample"}

    in_dim = int(W.shape[1])
    if weight_index is None:
        weight_index = int(G.detach().abs().flatten().argmax())
    i, j = divmod(int(weight_index), in_dim)

    w0 = float(W[i, j].detach())
    g = float(G[i, j].detach())
    lr, opt_type = _optimizer_lr(nodes)

    data_out = results.get(data_nid, {})
    orig = W.detach()[i, j].item()
    R = max(0.5, 2.0 * abs(w0))

    curve = []
    with torch.no_grad():
        for t in range(num_points):
            wv = w0 - R + (2.0 * R) * t / (num_points - 1)
            W[i, j] = wv
            pt = _eval_point(modules, nodes, edges, data_out, data_nid, loss_nid, pred_nid, true_label)
            curve.append({"w": _safe_float(wv), "loss": pt["loss"], "pTrue": pt["pTrue"], "pred": pt["pred"]})
        # One gradient step along this weight: w − lr·∂L/∂w
        w_step = w0 - lr * g
        W[i, j] = w_step
        loss_step = _eval_point(modules, nodes, edges, data_out, data_nid, loss_nid)["loss"]
        W[i, j] = orig  # restore — leave the stored model unchanged

    return {
        "nodeId": node_id,
        "displayName": _friendly_name(nodes[node_id]["type"]),
        "coord": [i, j],
        "outDim": int(W.shape[0]),
        "inDim": in_dim,
        "lr": lr,
        "optimizerType": opt_type,
        "sampleIdx": sample_idx,
        "current": {"w": _safe_float(w0), "loss": loss0, "grad": _safe_float(g)},
        "step": {"w": _safe_float(w_step), "loss": loss_step},
        "trueLabel": true_label,
        "classNames": class_names,
        "curve": curve,
    }
