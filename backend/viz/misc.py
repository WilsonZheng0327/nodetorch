"""Miscellaneous node visualizations: embedding, recurrent, attention, structural, default."""

import torch
from graph_builder import _safe_float
from .helpers import default_transformation, feature_maps_data, vector_data, histogram_data, EMPTY_FMAPS


def forward_viz_embedding(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "default"}
    if output is not None:
        transformation = default_transformation(output)
    insight = None
    if output is not None and output.dim() >= 2:
        insight = f"Looked up embeddings of dim {output.shape[-1]} for each token"
    return {"transformation": transformation, "insight": insight}


def forward_viz_recurrent(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "default"}
    if output is not None:
        transformation = default_transformation(output)
    return {"transformation": transformation, "insight": "Processed sequence through recurrent cells"}


MAX_ATTENTION_SIZE = 256   # cap the [seq x seq] heatmap so we don't ship huge grids
MAX_FOCUS_TOKENS = 256     # cap the focused-row bar chart (full-resolution)


def _row_entropy(row):
    """Shannon entropy of a probability row, normalized to [0, 1] where 1 = uniform."""
    import math
    s = 0.0
    n = 0
    for p in row:
        if p > 1e-12:
            s -= p * math.log(p)
            n += 1
    if n <= 1:
        return 0.0
    return s / math.log(n)


def _focus_row_data(weights, query_index, tokens):
    """Build a full-resolution attention distribution for one query position.

    weights: torch.Tensor [H, S, S] — already detached/cpu.
    Returns a dict with per-head + avg rows, top-k attended positions, and labels.
    """
    H, S, _ = weights.shape
    qi = max(0, min(S - 1, query_index))
    # If S is huge, only ship the focus row up to MAX_FOCUS_TOKENS.
    keep = min(S, MAX_FOCUS_TOKENS)
    head_rows = []
    top_per_head = []
    for h in range(H):
        full_row = weights[h, qi].tolist()
        row = [_safe_float(v) for v in full_row[:keep]]
        head_rows.append(row)
        # top-5 indices in the full row
        idxs = sorted(range(len(full_row)), key=lambda i: full_row[i], reverse=True)[:5]
        top_per_head.append([{"index": i, "weight": _safe_float(float(full_row[i]))} for i in idxs])
    # Average row
    avg_row = weights[:, qi].mean(dim=0).tolist()
    avg_full = [_safe_float(v) for v in avg_row[:keep]]
    avg_idxs = sorted(range(len(avg_row)), key=lambda i: avg_row[i], reverse=True)[:5]
    avg_top = [{"index": i, "weight": _safe_float(float(avg_row[i]))} for i in avg_idxs]

    labels = None
    if tokens is not None:
        labels = list(tokens[:keep])

    return {
        "queryIndex": qi,
        "queryToken": tokens[qi] if tokens is not None and qi < len(tokens) else None,
        "fullLen": S,
        "shownLen": keep,
        "perHeadRow": head_rows,
        "avgRow": avg_full,
        "perHeadTop": top_per_head,
        "avgTop": avg_top,
        "labels": labels,
    }


def _attention_transformation(weights, num_heads, is_causal, tokens=None):
    """Build an 'attention' transformation dict from per-head weights tensor [H, S, S]."""
    import torch.nn.functional as F
    H, S, _ = weights.shape

    # Downsample the matrices if seq_len is large (preserve average pattern).
    if S > MAX_ATTENTION_SIZE:
        target = MAX_ATTENTION_SIZE
        pooled = F.adaptive_avg_pool2d(weights.unsqueeze(1).float(), (target, target)).squeeze(1)
        pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        display = pooled
        display_size = target
    else:
        display = weights
        display_size = S

    avg = display.mean(dim=0)  # [S, S]

    # Per-head entropy (averaged across query positions) — low = sharp/focused, high = diffuse.
    head_entropy = []
    for h in range(num_heads):
        rows = weights[h].tolist()
        ents = [_row_entropy(r) for r in rows]
        head_entropy.append(_safe_float(sum(ents) / max(1, len(ents))))

    # Default focused row: last position (most relevant for autoregressive / next-token prediction).
    focus_row = _focus_row_data(weights, S - 1, tokens)

    # Truncate labels to displaySize for the heatmap-level axis labels (when seq small enough).
    heatmap_labels = None
    if tokens is not None and S == display_size:
        heatmap_labels = list(tokens[:S])

    return {
        "type": "attention",
        "numHeads": num_heads,
        "seqLen": S,
        "displaySize": display_size,
        "causalMask": bool(is_causal),
        "perHeadWeights": [[[_safe_float(v) for v in row] for row in head.tolist()] for head in display],
        "avgWeights": [[_safe_float(v) for v in row] for row in avg.tolist()],
        "headEntropy": head_entropy,
        "tokens": heatmap_labels,
        "focusRow": focus_row,
    }


def forward_viz_mha(node_type, module, input_tensor, output, inputs, out_dict):
    """Re-run the underlying nn.MultiheadAttention asking for per-head weights."""
    import torch
    transformation: dict = {"type": "default"}
    insight = "Attention mechanism: weighted sum over positions"
    tokens = out_dict.get("_sample_tokens") if isinstance(out_dict, dict) else None

    q = inputs.get("query"); k = inputs.get("key"); v = inputs.get("value")
    if (module is not None and hasattr(module, "mha")
            and isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)):
        try:
            mha = module.mha
            is_causal = bool(getattr(module, "is_causal", False))
            seq_len = q.shape[1]
            attn_mask = None
            if is_causal:
                attn_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
            with torch.no_grad():
                was_training = mha.training
                mha.eval()
                _, weights = mha(q, k, v, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
                if was_training:
                    mha.train()
            # weights: [B, H, S, S]
            if weights is not None and weights.dim() == 4:
                w0 = weights[0].detach().float().cpu()  # [H, S, S]
                transformation = _attention_transformation(
                    w0, num_heads=w0.shape[0], is_causal=is_causal, tokens=tokens,
                )
                insight = f"Multi-head attention across {w0.shape[0]} heads, {w0.shape[1]} positions"
                if is_causal:
                    insight += " — causal mask hides future positions"
        except Exception as e:
            import logging
            logging.getLogger("nodetorch").warning(f"MHA viz fell back to default: {e}")
            if output is not None:
                transformation = default_transformation(output)
    elif output is not None:
        transformation = default_transformation(output)

    return {"transformation": transformation, "insight": insight}


def forward_viz_sdpa(node_type, module, input_tensor, output, inputs, out_dict):
    """Manually compute softmax(QK^T / sqrt(d)) for the SDPA node (no built-in weights)."""
    import torch
    import math
    transformation: dict = {"type": "default"}
    insight = "Attention mechanism: weighted sum over positions"
    tokens = out_dict.get("_sample_tokens") if isinstance(out_dict, dict) else None

    q = inputs.get("query"); k = inputs.get("key"); v = inputs.get("value")
    is_causal = bool(getattr(module, "is_causal", False)) if module is not None else False

    if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
        try:
            with torch.no_grad():
                if q.dim() == 3:
                    d = q.shape[-1]
                    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # [B, S, S]
                    if is_causal:
                        S = scores.shape[-1]
                        mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
                        scores = scores.masked_fill(mask, float("-inf"))
                    weights = torch.softmax(scores, dim=-1)
                    w = weights[0].unsqueeze(0).detach().float().cpu()  # [1, S, S]
                    num_heads = 1
                elif q.dim() == 4:
                    d = q.shape[-1]
                    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # [B, H, S, S]
                    if is_causal:
                        S = scores.shape[-1]
                        mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
                        scores = scores.masked_fill(mask, float("-inf"))
                    weights = torch.softmax(scores, dim=-1)
                    w = weights[0].detach().float().cpu()  # [H, S, S]
                    num_heads = w.shape[0]
                else:
                    w = None
                    num_heads = 0

            if w is not None and w.numel() > 0:
                transformation = _attention_transformation(
                    w, num_heads=num_heads, is_causal=is_causal, tokens=tokens,
                )
                insight = f"Scaled dot-product attention over {w.shape[-1]} positions"
                if is_causal:
                    insight += " — causal mask hides future positions"
        except Exception as e:
            import logging
            logging.getLogger("nodetorch").warning(f"SDPA viz fell back to default: {e}")
            if output is not None:
                transformation = default_transformation(output)
    elif output is not None:
        transformation = default_transformation(output)

    return {"transformation": transformation, "insight": insight}


def forward_viz_add(node_type, module, input_tensor, output, inputs, out_dict):
    """Add: show all inputs and the element-wise sum."""
    transformation: dict = {"type": "add", "inputs": []}

    # Collect all inputs
    for port_id in sorted(inputs.keys()):
        t = inputs[port_id]
        if not isinstance(t, torch.Tensor):
            continue
        entry: dict = {"label": port_id.upper()}
        if t.dim() == 4:
            entry["featureMaps"] = feature_maps_data(t[0])
        elif t.dim() >= 1:
            entry["vector"] = vector_data(t[0] if t.dim() >= 2 else t)
        transformation["inputs"].append(entry)

    # Output
    if output is not None and isinstance(output, torch.Tensor):
        if output.dim() == 4:
            transformation["output"] = feature_maps_data(output[0])
        else:
            transformation["output"] = None
            transformation["outputVector"] = vector_data(output[0] if output.dim() >= 2 else output)
    else:
        transformation["output"] = None

    n_inputs = len(transformation["inputs"])
    return {"transformation": transformation, "insight": f"Element-wise sum of {n_inputs} inputs (residual / skip connection)"}


def forward_viz_reparameterize(node_type, module, input_tensor, output, inputs, out_dict):
    """VAE reparameterize: show mean + logvar inputs → sampled z output."""
    transformation: dict = {"type": "reparameterize"}

    mean_t = inputs.get("mean")
    logvar_t = inputs.get("logvar")

    if mean_t is not None and isinstance(mean_t, torch.Tensor):
        m = mean_t[0].detach().float().flatten() if mean_t.dim() >= 2 else mean_t.detach().float().flatten()
        transformation["meanValues"] = [_safe_float(float(v)) for v in m[:128].tolist()]
        transformation["meanHist"] = histogram_data(mean_t)
    if logvar_t is not None and isinstance(logvar_t, torch.Tensor):
        lv = logvar_t[0].detach().float().flatten() if logvar_t.dim() >= 2 else logvar_t.detach().float().flatten()
        transformation["logvarValues"] = [_safe_float(float(v)) for v in lv[:128].tolist()]
        transformation["logvarHist"] = histogram_data(logvar_t)
    if output is not None and isinstance(output, torch.Tensor):
        z = output[0].detach().float().flatten() if output.dim() >= 2 else output.detach().float().flatten()
        transformation["zValues"] = [_safe_float(float(v)) for v in z[:128].tolist()]
        transformation["latentDim"] = int(z.numel())

    return {
        "transformation": transformation,
        "insight": "z = mean + exp(0.5 \u00d7 logvar) \u00d7 noise \u2014 samples from the learned latent distribution",
    }


def forward_viz_reshape(node_type, module, input_tensor, output, inputs, out_dict):
    """Reshape: show before and after with shape labels."""
    transformation: dict = {"type": "reshape"}

    if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
        transformation["inputShape"] = list(input_tensor.shape)
        if input_tensor.dim() == 4:
            transformation["inputFmaps"] = feature_maps_data(input_tensor[0])
        elif input_tensor.dim() >= 1:
            transformation["inputVector"] = vector_data(input_tensor[0] if input_tensor.dim() >= 2 else input_tensor)
    else:
        transformation["inputShape"] = []

    if output is not None and isinstance(output, torch.Tensor):
        transformation["outputShape"] = list(output.shape)
        if output.dim() == 4:
            transformation["outputFmaps"] = feature_maps_data(output[0])
        elif output.dim() >= 1:
            transformation["outputVector"] = vector_data(output[0] if output.dim() >= 2 else output)
    else:
        transformation["outputShape"] = []

    in_s = transformation.get("inputShape", [])
    out_s = transformation.get("outputShape", [])
    return {
        "transformation": transformation,
        "insight": f"Reshaped {in_s} \u2192 {out_s} \u2014 same data, different layout",
    }


def _is_constant_tensor(t: torch.Tensor) -> tuple[bool, float]:
    """Check if a tensor has the same value everywhere."""
    flat = t.detach().float().flatten()
    if flat.numel() == 0:
        return False, 0.0
    val = float(flat[0])
    is_const = bool(torch.all(flat == val))
    return is_const, val


def _build_concat_entry(port_id: str, t: torch.Tensor) -> dict:
    """Build a concat input/output entry with smart visualization."""
    entry: dict = {"label": port_id, "shape": list(t.shape)}

    # Check if it's a constant (like timestep channel)
    is_const, const_val = _is_constant_tensor(t)
    if is_const:
        entry["isConstant"] = True
        entry["constantValue"] = _safe_float(const_val)
        return entry

    # Normal tensor
    if t.dim() == 4:
        entry["featureMaps"] = feature_maps_data(t[0])
    elif t.dim() == 3:
        entry["vector"] = vector_data(t[0].flatten())
    elif t.dim() >= 1:
        entry["vector"] = vector_data(t[0] if t.dim() >= 2 else t)
    return entry


def forward_viz_concat(node_type, module, input_tensor, output, inputs, out_dict):
    """Concat: show all inputs as columns + the concatenated output."""
    transformation: dict = {"type": "concat", "inputs": []}

    for port_id in sorted(inputs.keys()):
        t = inputs[port_id]
        if not isinstance(t, torch.Tensor):
            continue
        entry = _build_concat_entry(port_id.replace("in_", "Input "), t)
        transformation["inputs"].append(entry)

    # Output
    if output is not None and isinstance(output, torch.Tensor):
        transformation["outputShape"] = list(output.shape)
        if output.dim() == 4:
            transformation["outputFmaps"] = feature_maps_data(output[0])
        elif output.dim() == 3:
            transformation["outputVector"] = vector_data(output[0].flatten())
        else:
            transformation["outputVector"] = vector_data(output[0] if output.dim() >= 2 else output)

    # Get concat dimension
    dim = 1
    if module is not None and hasattr(module, 'dim'):
        dim = module.dim
    transformation["dim"] = dim

    n = len(transformation["inputs"])
    return {"transformation": transformation, "insight": f"Concatenated {n} inputs along dimension {dim}"}


def forward_viz_structural(node_type, module, input_tensor, output, inputs, out_dict):
    if node_type == "ml.structural.add":
        return forward_viz_add(node_type, module, input_tensor, output, inputs, out_dict)
    if node_type == "ml.structural.concat":
        return forward_viz_concat(node_type, module, input_tensor, output, inputs, out_dict)
    if node_type == "ml.structural.reparameterize":
        return forward_viz_reparameterize(node_type, module, input_tensor, output, inputs, out_dict)
    if node_type == "ml.structural.reshape":
        return forward_viz_reshape(node_type, module, input_tensor, output, inputs, out_dict)

    insights = {
        "ml.structural.permute": "Reordered tensor dimensions",
        "ml.structural.sequence_pool": "Pooled sequence to single vector per sample",
    }
    transformation: dict = {"type": "default"}
    if output is not None:
        transformation = default_transformation(output)
    insight = None
    if node_type in insights:
        insight = insights[node_type]
    return {"transformation": transformation, "insight": insight}


def forward_viz_default(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "default"}
    if output is not None and isinstance(output, torch.Tensor):
        transformation = default_transformation(output)
    return {"transformation": transformation}
