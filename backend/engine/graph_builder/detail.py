"""Per-node detail visualization for the inspector modal.

``get_layer_detail`` reuses the cached ``_last_run`` (or re-runs a forward
pass) to produce weight heatmaps, conv kernels/feature maps, attention maps,
LSTM/GRU hidden state, and loss-node confusion matrices for a single node.
"""

import torch

from dataprep.data_loaders import DATA_LOADERS, CLASS_NAMES
from engine.graph_builder.constants import LOSS_NODES
from engine.graph_builder._state import _last_run
from engine.graph_builder.build import gather_inputs, SubGraphModule
from engine.graph_builder.forward import build_and_run_graph
from engine.graph_builder.stats import _safe_float


def get_layer_detail(graph_data: dict, node_id: str) -> dict:
    """Return detailed visualization data for a specific node.

    Uses cached results from the last forward/train/infer pass.
    Falls back to running a forward pass if no cache exists.

    Returns:
    - weightMatrix: 2D weight data for heatmap (Linear, Conv2d)
    - featureMaps: per-channel activation grids (Conv2d)
    - attentionMap: attention weight matrix (MHA, Attention)
    - hiddenState: hidden/cell state matrix (LSTM, GRU)
    - confusionData: predictions vs labels (loss nodes)
    """
    # Use cached results if available, otherwise run a forward pass
    if _last_run:
        modules = _last_run["modules"]
        results = _last_run["results"]
        nodes = _last_run["nodes"]
        edges = _last_run["edges"]
    else:
        with torch.no_grad():
            modules, results, _, nodes, edges = build_and_run_graph(graph_data)

    # Resolve module — might be inside a subgraph
    module = modules.get(node_id)
    output = results.get(node_id, {}).get("out")
    node = nodes.get(node_id)

    # Check if node is inside a subgraph
    if not node:
        for nid, mod in modules.items():
            if isinstance(mod, SubGraphModule) and node_id in mod._key_map:
                safe_key = mod._key_map[node_id]
                if safe_key in mod.inner_modules:
                    module = mod.inner_modules[safe_key]
                inner_results = getattr(mod, '_last_results', {})
                output = inner_results.get(node_id, {}).get("out")
                node = mod.inner_nodes.get(node_id)
                break

    if not node:
        return {"error": f"Node {node_id} not found"}

    node_type = node["type"]
    detail: dict = {"nodeType": node_type}

    # --- Weight visualization ---
    if module is not None:
        weight = None
        for name, param in module.named_parameters():
            if 'weight' in name:
                weight = param.detach().cpu().float()
                break
        if weight is not None:
            if weight.dim() == 4:
                # Conv2d: [out_ch, in_ch, kH, kW] — show as kernel grid
                out_ch, in_ch, kH, kW = weight.shape
                n_filters = min(32, out_ch)
                kernels = []
                for f in range(n_filters):
                    # Average across input channels to get one kH x kW image per filter
                    kernel = weight[f].mean(dim=0)  # [kH, kW]
                    # Normalize to 0-255
                    kmin, kmax = float(kernel.min()), float(kernel.max())
                    rng = kmax - kmin if kmax != kmin else 1.0
                    normalized = ((kernel - kmin) / rng * 255).clamp(0, 255).byte()
                    kernels.append(normalized.tolist())
                detail["convKernels"] = {
                    "kernels": kernels,
                    "count": n_filters,
                    "totalFilters": out_ch,
                    "height": kH,
                    "width": kW,
                    "inChannels": in_ch,
                }
            else:
                # Linear or other: 2D weight matrix heatmap
                if weight.dim() == 1:
                    mat = weight.unsqueeze(0)
                elif weight.dim() == 2:
                    mat = weight
                else:
                    mat = weight.reshape(weight.shape[0], -1)

                actual_rows, actual_cols = mat.shape[0], mat.shape[1]
                vmin = _safe_float(float(mat.min()))
                vmax = _safe_float(float(mat.max()))

                # Downsample by block-averaging if too large (max 128x128 for display)
                MAX_DIM = 128
                if mat.shape[0] > MAX_DIM or mat.shape[1] > MAX_DIM:
                    mat = torch.nn.functional.interpolate(
                        mat.unsqueeze(0).unsqueeze(0),
                        size=(min(MAX_DIM, mat.shape[0]), min(MAX_DIM, mat.shape[1])),
                        mode='area',
                    ).squeeze()

                detail["weightMatrix"] = {
                    "data": mat.tolist(),
                    "rows": mat.shape[0],
                    "cols": mat.shape[1],
                    "actualRows": actual_rows,
                    "actualCols": actual_cols,
                    "min": vmin,
                    "max": vmax,
                }

    # --- Feature maps (Conv output channels) ---
    # Only show for layers that actually *produce* new feature channels via learned
    # filters. ReLU, Pool, BatchNorm etc. pass 4D tensors through but don't create
    # new features — labeling their output "feature maps" would mislead students.
    FEATURE_MAP_TYPES = {
        "ml.layers.conv2d",
        "ml.layers.conv_transpose2d",
        "ml.layers.pretrained_resnet18",
    }
    if node_type in FEATURE_MAP_TYPES and output is not None and isinstance(output, torch.Tensor) and output.dim() == 4:
        # [batch, channels, H, W] → take first sample, up to 16 channels
        fmaps = output[0].detach().cpu().float()
        n_maps = min(16, fmaps.shape[0])
        maps_list = []
        for c in range(n_maps):
            fm = fmaps[c]
            # Normalize to 0-255
            fmin, fmax = float(fm.min()), float(fm.max())
            rng = fmax - fmin if fmax != fmin else 1.0
            normalized = ((fm - fmin) / rng * 255).clamp(0, 255).byte()
            # Downsample if larger than 32x32
            if normalized.shape[0] > 32 or normalized.shape[1] > 32:
                normalized = torch.nn.functional.interpolate(
                    normalized.unsqueeze(0).unsqueeze(0).float(),
                    size=(min(32, normalized.shape[0]), min(32, normalized.shape[1])),
                    mode='nearest',
                ).squeeze().byte()
            maps_list.append(normalized.tolist())
        detail["featureMaps"] = {
            "maps": maps_list,
            "channels": n_maps,
            "height": len(maps_list[0]),
            "width": len(maps_list[0][0]) if maps_list[0] else 0,
        }

    # --- Attention map ---
    # Re-run MHA/Attention to capture attention weights
    if node_type in ("ml.layers.multihead_attention", "ml.layers.attention"):
        if module is not None:
            inputs = gather_inputs(node_id, edges, results)
            # MHA/Attention use query/key/value ports, not "in"
            query = inputs.get("query")
            if query is not None:
                key = inputs.get("key", query)
                value = inputs.get("value", query)
                try:
                    if node_type == "ml.layers.multihead_attention":
                        kwargs = {"need_weights": True, "average_attn_weights": True}
                        if getattr(module, "is_causal", False):
                            seq_len = query.shape[1]
                            kwargs["attn_mask"] = torch.triu(
                                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                                diagonal=1,
                            )
                        _, attn_weights = module.mha(query, key, value, **kwargs)
                    else:
                        # SDPA: compute attention weights manually
                        import math as _math
                        d_k = query.shape[-1]
                        scores = torch.matmul(query, key.transpose(-2, -1)) / _math.sqrt(d_k)
                        attn_weights = torch.softmax(scores, dim=-1)
                    if attn_weights is not None:
                        # Take first sample, first head
                        am = attn_weights[0]
                        if am.dim() == 3:
                            am = am[0]  # first head
                        am = am.detach().cpu().float()
                        if am.shape[0] > 64:
                            am = am[:64, :64]
                        detail["attentionMap"] = {
                            "data": am.tolist(),
                            "rows": am.shape[0],
                            "cols": am.shape[1],
                        }
                except Exception:
                    pass  # Attention capture failed, skip

    # --- Hidden state (LSTM/GRU) ---
    if node_type in ("ml.layers.lstm", "ml.layers.gru"):
        node_output = results.get(node_id, {})
        hidden = node_output.get("hidden")
        if hidden is not None and isinstance(hidden, torch.Tensor):
            # [num_layers, batch, hidden_size] → take first sample
            h = hidden[0, 0].detach().cpu().float() if hidden.dim() == 3 else hidden[0].detach().cpu().float()
            detail["hiddenState"] = {
                "data": h.unsqueeze(0).tolist(),
                "rows": 1,
                "cols": len(h),
                "label": "Hidden State",
            }
        cell = node_output.get("cell")
        if cell is not None and isinstance(cell, torch.Tensor):
            c = cell[0, 0].detach().cpu().float() if cell.dim() == 3 else cell[0].detach().cpu().float()
            detail["cellState"] = {
                "data": c.unsqueeze(0).tolist(),
                "rows": 1,
                "cols": len(c),
                "label": "Cell State",
            }

    # --- Confusion matrix + misclassifications (loss nodes) ---
    if node_type in LOSS_NODES:
        # Use full confusion matrix accumulated during training if available
        cached_cm = _last_run.get("confusionMatrix")
        if cached_cm:
            detail["confusionMatrix"] = cached_cm
        misclass = _last_run.get("misclassifications")
        if misclass:
            detail["misclassifications"] = misclass
        else:
            # Fallback: compute from current batch in results
            pred_tensor = None
            label_tensor = None
            for edge in edges:
                if edge["target"]["nodeId"] == node_id:
                    if edge["target"]["portId"] == "predictions":
                        pred_tensor = results.get(edge["source"]["nodeId"], {}).get("out")
                    elif edge["target"]["portId"] == "labels":
                        label_tensor = results.get(edge["source"]["nodeId"], {}).get("labels")
                        if label_tensor is None:
                            label_tensor = results.get(edge["source"]["nodeId"], {}).get("out")
            if pred_tensor is not None and label_tensor is not None and pred_tensor.dim() == 2:
                preds = pred_tensor.argmax(dim=1).cpu()
                labels = label_tensor.cpu()
                n_classes = max(int(preds.max().item()) + 1, int(labels.max().item()) + 1)
                n_classes = max(n_classes, 2)
                matrix = [[0] * n_classes for _ in range(n_classes)]
                for p, l in zip(preds.tolist(), labels.tolist()):
                    if 0 <= p < n_classes and 0 <= l < n_classes:
                        matrix[l][p] += 1
                from dataprep.data_loaders import CLASS_NAMES
                data_type = None
                for nid, nd in _last_run.get("nodes", {}).items():
                    if nd.get("type", "") in DATA_LOADERS:
                        data_type = nd["type"]
                        break
                detail["confusionMatrix"] = {
                    "data": matrix,
                    "size": n_classes,
                    "classNames": CLASS_NAMES.get(data_type, []) if data_type else [],
                }

    return detail
