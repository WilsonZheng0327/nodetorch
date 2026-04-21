# export_python.py — Generate a standalone, runnable Python file from a NodeTorch graph.
#
# The generated file includes:
#   - All necessary imports
#   - Model class (nn.Module) with __init__ and forward
#   - Dataset loading with correct normalization
#   - Training loop (standard, VAE, GAN, or diffusion depending on graph)
#   - Device selection (CUDA/MPS/CPU)
#
# Usage:
#   from export_python import export_to_python
#   code = export_to_python(graph_data)
#   # code is a complete, runnable .py file as a string

import re
from graph_builder import (
    topological_sort, build_modules, gather_inputs,
    LOSS_NODES, ALL_LOSS_NODES, OPTIMIZER_NODES, MULTI_INPUT_NODES,
    GAN_NOISE_TYPE, DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE,
    SUBGRAPH_TYPE, SENTINEL_INPUT, SENTINEL_OUTPUT,
)
from training import detect_training_mode
from data_loaders import DATA_LOADERS


# ─── Dataset constants ────────────────────────────────────────────────────────

DATASET_CODE = {
    "data.mnist": {
        "class": "torchvision.datasets.MNIST",
        "norm_mean": "(0.1307,)",
        "norm_std": "(0.3081,)",
        "channels": 1,
        "classes": 10,
        "image_size": 28,
    },
    "data.cifar10": {
        "class": "torchvision.datasets.CIFAR10",
        "norm_mean": "(0.4914, 0.4822, 0.4465)",
        "norm_std": "(0.2470, 0.2435, 0.2616)",
        "channels": 3,
        "classes": 10,
        "image_size": 32,
    },
    "data.cifar100": {
        "class": "torchvision.datasets.CIFAR100",
        "norm_mean": "(0.5071, 0.4867, 0.4408)",
        "norm_std": "(0.2675, 0.2565, 0.2761)",
        "channels": 3,
        "classes": 100,
        "image_size": 32,
    },
    "data.fashion_mnist": {
        "class": "torchvision.datasets.FashionMNIST",
        "norm_mean": "(0.2860,)",
        "norm_std": "(0.3530,)",
        "channels": 1,
        "classes": 10,
        "image_size": 28,
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize_id(node_id: str) -> str:
    """Convert node ID to a valid Python identifier (for variable / attribute names)."""
    s = re.sub(r'[^a-zA-Z0-9_]', '_', node_id)
    if s[0:1].isdigit():
        s = '_' + s
    return s


def _indent(code: str, level: int = 2) -> str:
    """Indent every line of code by level * 4 spaces."""
    prefix = '    ' * level
    return '\n'.join(prefix + line if line.strip() else '' for line in code.split('\n'))


def _get_optimizer_props(nodes: dict) -> dict:
    """Find the first optimizer node and return its properties."""
    for n in nodes.values():
        if n["type"] in OPTIMIZER_NODES:
            return n.get("properties", {})
    return {"lr": 0.001, "epochs": 10}


def _get_optimizer_type(nodes: dict) -> str:
    """Return the optimizer node type string."""
    for n in nodes.values():
        if n["type"] in OPTIMIZER_NODES:
            return n["type"]
    return "ml.optimizers.adam"


def _get_data_node(nodes: dict) -> dict | None:
    """Find the data node."""
    for n in nodes.values():
        if n["type"] in DATA_LOADERS:
            return n
    return None


def _get_loss_node(nodes: dict) -> dict | None:
    """Find the loss node."""
    for n in nodes.values():
        if n["type"] in ALL_LOSS_NODES:
            return n
    return None


# ─── Module code generation ───────────────────────────────────────────────────

def _module_code(node_type: str, props: dict, module) -> str | None:
    """Generate nn.Module constructor code for a given node type.

    Uses the actual built module to resolve input dimensions (in_channels, in_features, etc.).
    Returns None for nodes that don't become modules (data, loss, optimizer, etc.).
    """
    if node_type == "ml.layers.conv2d":
        m = module
        return f"nn.Conv2d({m.in_channels}, {m.out_channels}, kernel_size={m.kernel_size[0]}, stride={m.stride[0]}, padding={m.padding[0]})"

    if node_type == "ml.layers.conv1d":
        m = module
        return f"nn.Conv1d({m.in_channels}, {m.out_channels}, kernel_size={m.kernel_size[0]}, stride={m.stride[0]}, padding={m.padding[0]})"

    if node_type == "ml.layers.conv_transpose2d":
        m = module
        op = m.output_padding[0] if hasattr(m, 'output_padding') else 0
        return f"nn.ConvTranspose2d({m.in_channels}, {m.out_channels}, kernel_size={m.kernel_size[0]}, stride={m.stride[0]}, padding={m.padding[0]}, output_padding={op})"

    if node_type == "ml.layers.linear":
        m = module
        return f"nn.Linear({m.in_features}, {m.out_features})"

    if node_type == "ml.layers.flatten":
        return "nn.Flatten()"

    if node_type == "ml.layers.maxpool2d":
        ks = props["kernelSize"]
        st = props["stride"]
        pa = props["padding"]
        return f"nn.MaxPool2d(kernel_size={ks}, stride={st}, padding={pa})"

    if node_type == "ml.layers.maxpool1d":
        ks = props["kernelSize"]
        st = props["stride"]
        pa = props["padding"]
        return f"nn.MaxPool1d(kernel_size={ks}, stride={st}, padding={pa})"

    if node_type == "ml.layers.avgpool2d":
        ks = props["kernelSize"]
        st = props["stride"]
        pa = props["padding"]
        return f"nn.AvgPool2d(kernel_size={ks}, stride={st}, padding={pa})"

    if node_type == "ml.layers.adaptive_avgpool2d":
        oh = props["outputH"]
        ow = props["outputW"]
        return f"nn.AdaptiveAvgPool2d(({oh}, {ow}))"

    if node_type == "ml.layers.batchnorm2d":
        m = module
        return f"nn.BatchNorm2d({m.num_features})"

    if node_type == "ml.layers.batchnorm1d":
        m = module
        return f"nn.BatchNorm1d({m.num_features})"

    if node_type == "ml.layers.groupnorm":
        m = module
        return f"nn.GroupNorm({m.num_groups}, {m.num_channels})"

    if node_type == "ml.layers.instancenorm2d":
        m = module
        return f"nn.InstanceNorm2d({m.num_features}, affine=True)"

    if node_type == "ml.layers.dropout":
        p = props["p"]
        return f"nn.Dropout(p={p})"

    if node_type == "ml.layers.dropout2d":
        p = props["p"]
        return f"nn.Dropout2d(p={p})"

    if node_type == "ml.layers.layernorm":
        m = module
        ns = list(m.normalized_shape)
        return f"nn.LayerNorm({ns})"

    if node_type == "ml.layers.embedding":
        ne = props["numEmbeddings"]
        ed = props["embeddingDim"]
        return f"nn.Embedding({ne}, {ed})"

    if node_type == "ml.layers.upsample":
        sf = props["scaleFactor"]
        mode = props.get("mode", "nearest")
        return f"nn.Upsample(scale_factor={sf}, mode='{mode}')"

    if node_type == "ml.layers.lstm":
        m = module.lstm  # LSTMWrapper
        return f"nn.LSTM(input_size={m.input_size}, hidden_size={m.hidden_size}, num_layers={m.num_layers}, batch_first=True, bidirectional={m.bidirectional})"

    if node_type == "ml.layers.gru":
        m = module.gru  # GRUWrapper
        return f"nn.GRU(input_size={m.input_size}, hidden_size={m.hidden_size}, num_layers={m.num_layers}, batch_first=True, bidirectional={m.bidirectional})"

    if node_type == "ml.layers.rnn":
        m = module.rnn  # RNNWrapper
        return f"nn.RNN(input_size={m.input_size}, hidden_size={m.hidden_size}, num_layers={m.num_layers}, batch_first=True, bidirectional={m.bidirectional})"

    if node_type == "ml.layers.multihead_attention":
        m = module.mha  # MHAWrapper
        return f"nn.MultiheadAttention(embed_dim={m.embed_dim}, num_heads={m.num_heads}, batch_first=True)"

    if node_type == "ml.layers.pretrained_resnet18":
        mode = props.get("mode", "features")
        freeze = props.get("freeze", True)
        # This gets special handling — not a simple nn.Module line
        return f"__PRETRAINED_RESNET18__(mode='{mode}', freeze={freeze})"

    # Activations
    if node_type == "ml.activations.relu":
        return "nn.ReLU()"
    if node_type == "ml.activations.sigmoid":
        return "nn.Sigmoid()"
    if node_type == "ml.activations.tanh":
        return "nn.Tanh()"
    if node_type == "ml.activations.gelu":
        return "nn.GELU()"
    if node_type == "ml.activations.leaky_relu":
        ns = props.get("negativeSlope", 0.01)
        return f"nn.LeakyReLU(negative_slope={ns})"
    if node_type == "ml.activations.softmax":
        dim = props.get("dim", -1)
        return f"nn.Softmax(dim={dim})"

    return None


# ─── Inline operation code (for nodes that don't become modules) ──────────────

def _inline_code(node_type: str, props: dict) -> str | None:
    """Return inline forward code for nodes that don't become self.<name> modules."""
    if node_type == "ml.structural.reshape":
        target_str = props.get("targetShape", "-1")
        parts = [s.strip() for s in target_str.split(",")]
        # Replace leading -1 with x.size(0) for batch dim
        if parts[0] == "-1":
            parts[0] = "x.size(0)"
        shape_str = ", ".join(parts)
        return f"x = x.reshape({shape_str})"

    if node_type == "ml.structural.permute":
        dims_str = props.get("dims", "0, 2, 1")
        return f"x = x.permute({dims_str})"

    if node_type == "ml.structural.sequence_pool":
        mode = props.get("mode", "last")
        if mode == "last":
            return "x = x[:, -1, :]"
        elif mode == "mean":
            return "x = x.mean(dim=1)"
        elif mode == "max":
            return "x = x.max(dim=1).values"
        return "x = x[:, -1, :]"

    return None


# ─── Node classification ──────────────────────────────────────────────────────

INLINE_NODES = {"ml.structural.reshape", "ml.structural.permute", "ml.structural.sequence_pool"}
SKIP_NODES = set(ALL_LOSS_NODES) | set(OPTIMIZER_NODES) | set(DATA_LOADERS.keys()) | {GAN_NOISE_TYPE, DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE}
MULTI_OUTPUT_NODES = {"ml.layers.lstm", "ml.layers.gru", "ml.layers.rnn"}


# ─── Forward body generation ─────────────────────────────────────────────────

def _build_forward_body(
    order: list[str],
    nodes: dict,
    edges: list,
    modules: dict,
    model_nodes: set[str],
    class_prefix: str = "",
) -> str:
    """Generate the forward() method body for a model.

    Handles:
    - Sequential flow (x = self.layer(x))
    - Fan-out (named variables for nodes whose output feeds multiple downstream)
    - Skip connections / Add nodes
    - Concat nodes (multiple named inputs)
    - Reparameterize (mean + logvar)
    - LSTM/GRU tuple unpacking
    """
    lines: list[str] = []

    # Precompute: for each node, which downstream nodes consume its output?
    downstream_consumers: dict[str, list[str]] = {nid: [] for nid in order}
    # Also track: for each target node, which source nodes feed it (and via which ports)?
    upstream_map: dict[str, list[tuple[str, str, str]]] = {nid: [] for nid in order}
    # (source_node_id, source_port_id, target_port_id)

    for edge in edges:
        src_id = edge["source"]["nodeId"]
        tgt_id = edge["target"]["nodeId"]
        if src_id in downstream_consumers:
            downstream_consumers[src_id].append(tgt_id)
        if tgt_id in upstream_map:
            upstream_map[tgt_id].append((src_id, edge["source"]["portId"], edge["target"]["portId"]))

    # Determine which nodes need named variables (fan-out or non-sequential access)
    # A node needs a named var if:
    # 1. Its output feeds >1 downstream model nodes
    # 2. Its output is consumed by an Add/Concat/Reparameterize node (not via "in" port)
    # 3. Its consumer is not the immediately-next model node in topological order
    #    (i.e., there are intervening model nodes that would overwrite 'x')
    needs_var: set[str] = set()

    # Build ordered list of model nodes only
    model_order = [nid for nid in order if nid in model_nodes]
    model_index = {nid: i for i, nid in enumerate(model_order)}

    for nid in order:
        if nid not in model_nodes:
            continue
        consumers = [c for c in downstream_consumers[nid] if c in model_nodes]
        if len(consumers) > 1:
            needs_var.add(nid)
        # If any consumer is a multi-input node, mark source for named var
        for c in consumers:
            c_node = nodes[c]
            if c_node["type"] in MULTI_INPUT_NODES:
                needs_var.add(nid)
        # If consumer is NOT the immediate next model node, we need a named var
        # because other model nodes will overwrite 'x' in between.
        nid_idx = model_index.get(nid)
        if nid_idx is not None:
            for c in consumers:
                c_idx = model_index.get(c)
                if c_idx is not None and c_idx > nid_idx + 1:
                    needs_var.add(nid)
                    break

    # Detect if the function input (x) fans out — i.e., non-model nodes feed
    # multiple model nodes. In this case we must preserve 'x' as 'x_input'.
    input_fans_out = False
    input_fanout_consumers: set[str] = set()
    for nid in order:
        if nid in model_nodes:
            continue
        consumers = [c for c in downstream_consumers.get(nid, []) if c in model_nodes]
        if len(consumers) > 1:
            input_fans_out = True
            input_fanout_consumers.update(consumers)

    # When the input fans out, all model nodes that directly read from the
    # non-model source need named variables (to avoid overwriting each other's results)
    if input_fans_out:
        for nid in input_fanout_consumers:
            needs_var.add(nid)

    # Variable name for each node's output
    var_names: dict[str, str] = {}
    for nid in order:
        if nid in needs_var:
            var_names[nid] = _sanitize_id(nid)

    # Track which VAE loss inputs we need to return
    vae_loss_node = None
    vae_inputs: dict[str, str] = {}  # port -> var name
    for nid in order:
        n = nodes[nid]
        if n["type"] == "ml.loss.vae":
            vae_loss_node = nid
            for src_id, src_port, tgt_port in upstream_map[nid]:
                if src_id in model_nodes:
                    vae_inputs[tgt_port] = var_names.get(src_id, "x")

    # Current "active" variable (x by default)
    last_var = "x"

    # If the function input fans out, save it before any layer modifies x
    if input_fans_out:
        lines.append("x_input = x")

    for nid in order:
        node = nodes[nid]
        ntype = node["type"]

        # Skip non-model nodes
        if nid not in model_nodes:
            continue

        attr_name = f"self.{class_prefix}{_sanitize_id(nid)}"

        # --- Inline nodes (reshape, permute, sequence_pool) ---
        if ntype in INLINE_NODES:
            # Determine input variable
            input_sources = upstream_map.get(nid, [])
            model_sources = [(s, sp, tp) for s, sp, tp in input_sources if s in model_nodes]
            if model_sources:
                src_id = model_sources[0][0]
                input_var = var_names.get(src_id, last_var)
            else:
                input_var = "x_input" if input_fans_out else "x"

            inline = _inline_code(ntype, node.get("properties", {}))
            if inline:
                # Replace 'x' with the actual input variable if different
                if input_var != "x" and input_var != last_var:
                    inline = inline.replace("x = x.", f"x = {input_var}.")
                    inline = inline.replace("x = x[", f"x = {input_var}[")
                    inline = inline.replace("x.reshape(", f"{input_var}.reshape(")
                    inline = inline.replace("x.permute(", f"{input_var}.permute(")
                    inline = inline.replace("x.size(0)", f"{input_var}.size(0)")
                elif input_var == last_var and input_var != "x":
                    inline = inline.replace("x = x.", f"x = {input_var}.")
                    inline = inline.replace("x = x[", f"x = {input_var}[")
                    inline = inline.replace("x.reshape(", f"{input_var}.reshape(")
                    inline = inline.replace("x.permute(", f"{input_var}.permute(")
                    inline = inline.replace("x.size(0)", f"{input_var}.size(0)")

                if nid in needs_var:
                    vname = var_names[nid]
                    inline = inline.replace("x = ", f"{vname} = ", 1)
                    lines.append(inline)
                    last_var = vname
                else:
                    lines.append(inline)
                    last_var = "x"
            continue

        # --- Add node ---
        if ntype == "ml.structural.add":
            input_sources = upstream_map.get(nid, [])
            # Resolve both inputs (may come from model or non-model nodes)
            all_sources = sorted(input_sources, key=lambda x: x[2])  # sort by port (a, b)
            input_default = "x_input" if input_fans_out else "x"

            def _resolve_add_var(src_id):
                if src_id in model_nodes:
                    return var_names.get(src_id, last_var)
                return input_default

            if len(all_sources) >= 2:
                var_a = _resolve_add_var(all_sources[0][0])
                var_b = _resolve_add_var(all_sources[1][0])
                if nid in needs_var:
                    vname = var_names[nid]
                    lines.append(f"{vname} = {var_a} + {var_b}")
                    last_var = vname
                else:
                    lines.append(f"x = {var_a} + {var_b}")
                    last_var = "x"
            continue

        # --- Concat node ---
        if ntype == "ml.structural.concat":
            dim = node.get("properties", {}).get("dim", 1)
            input_sources = upstream_map.get(nid, [])
            model_sources = [(s, sp, tp) for s, sp, tp in input_sources if s in model_nodes]
            model_sources.sort(key=lambda x: x[2])  # Sort by target port (in_0, in_1, ...)
            vars_list = [var_names.get(s, last_var) for s, sp, tp in model_sources]
            joined = ", ".join(vars_list)
            if nid in needs_var:
                vname = var_names[nid]
                lines.append(f"{vname} = torch.cat([{joined}], dim={dim})")
                last_var = vname
            else:
                lines.append(f"x = torch.cat([{joined}], dim={dim})")
                last_var = "x"
            continue

        # --- Reparameterize node ---
        if ntype == "ml.structural.reparameterize":
            input_sources = upstream_map.get(nid, [])
            mean_var = "x"
            logvar_var = "x"
            for s, sp, tp in input_sources:
                if tp == "mean" and s in model_nodes:
                    mean_var = var_names.get(s, "mean")
                elif tp == "logvar" and s in model_nodes:
                    logvar_var = var_names.get(s, "logvar")
            if nid in needs_var:
                vname = var_names[nid]
                lines.append(f"std = torch.exp(0.5 * {logvar_var})")
                lines.append(f"eps = torch.randn_like(std)")
                lines.append(f"{vname} = {mean_var} + eps * std")
                last_var = vname
            else:
                lines.append(f"std = torch.exp(0.5 * {logvar_var})")
                lines.append(f"eps = torch.randn_like(std)")
                lines.append(f"x = {mean_var} + eps * std")
                last_var = "x"
            continue

        # --- Multi-head attention ---
        if ntype == "ml.layers.multihead_attention":
            input_sources = upstream_map.get(nid, [])
            # MHA uses query/key/value ports or just "in" for self-attention
            q_var = k_var = v_var = last_var
            for s, sp, tp in input_sources:
                if s in model_nodes:
                    v = var_names.get(s, last_var)
                    if tp == "query":
                        q_var = v
                    elif tp == "key":
                        k_var = v
                    elif tp == "value":
                        v_var = v
                    elif tp == "in":
                        q_var = k_var = v_var = v
            if nid in needs_var:
                vname = var_names[nid]
                lines.append(f"{vname}, _ = {attr_name}({q_var}, {k_var}, {v_var})")
                last_var = vname
            else:
                lines.append(f"x, _ = {attr_name}({q_var}, {k_var}, {v_var})")
                last_var = "x"
            continue

        # --- Attention (scaled dot product) ---
        if ntype == "ml.layers.attention":
            input_sources = upstream_map.get(nid, [])
            q_var = k_var = v_var = last_var
            for s, sp, tp in input_sources:
                if s in model_nodes:
                    v = var_names.get(s, last_var)
                    if tp == "query":
                        q_var = v
                    elif tp == "key":
                        k_var = v
                    elif tp == "value":
                        v_var = v
                    elif tp == "in":
                        q_var = k_var = v_var = v
            if nid in needs_var:
                vname = var_names[nid]
                lines.append(f"{vname} = torch.nn.functional.scaled_dot_product_attention({q_var}, {k_var}, {v_var})")
                last_var = vname
            else:
                lines.append(f"x = torch.nn.functional.scaled_dot_product_attention({q_var}, {k_var}, {v_var})")
                last_var = "x"
            continue

        # --- Standard single-input module ---
        # Determine input variable
        input_sources = upstream_map.get(nid, [])
        model_sources = [(s, sp, tp) for s, sp, tp in input_sources if s in model_nodes]
        if model_sources:
            src_id = model_sources[0][0]
            input_var = var_names.get(src_id, last_var)
        else:
            input_var = "x_input" if input_fans_out else "x"

        # LSTM/GRU: unpack tuple
        if ntype in MULTI_OUTPUT_NODES:
            if nid in needs_var:
                vname = var_names[nid]
                lines.append(f"{vname}, _ = {attr_name}({input_var})")
                last_var = vname
            else:
                lines.append(f"x, _ = {attr_name}({input_var})")
                last_var = "x"
            continue

        # Regular module call
        if nid in needs_var:
            vname = var_names[nid]
            lines.append(f"{vname} = {attr_name}({input_var})")
            last_var = vname
        else:
            lines.append(f"x = {attr_name}({input_var})")
            last_var = "x"

    # For VAE: return (reconstruction, mean, logvar)
    if vae_loss_node:
        recon_var = vae_inputs.get("reconstruction", last_var)
        mean_var = vae_inputs.get("mean", "mean")
        logvar_var = vae_inputs.get("logvar", "logvar")
        lines.append(f"return {recon_var}, {mean_var}, {logvar_var}")
    else:
        lines.append(f"return {last_var}")

    return '\n'.join('        ' + line for line in lines)


# ─── Subgraph (block) class generation ────────────────────────────────────────

def _generate_subgraph_class(node: dict, modules: dict, parent_modules: dict) -> str:
    """Generate an nn.Module class for a subgraph block."""
    subgraph = node["subgraph"]
    block_name = node.get("properties", {}).get("blockName", "Block")
    class_name = re.sub(r'[^a-zA-Z0-9]', '', block_name)
    if not class_name:
        class_name = "Block"

    sg_nodes = {n["id"]: n for n in subgraph["nodes"]}
    sg_edges = subgraph["edges"]
    sg_order = topological_sort(sg_nodes, sg_edges)

    # Get the SubGraphModule to access its inner modules
    sg_module = parent_modules.get(node["id"])
    if sg_module is None:
        return f"# Could not build {class_name} — module not available\n"

    # Collect model nodes (skip sentinels, loss, optimizer)
    model_nodes: set[str] = set()
    init_lines: list[str] = []

    for nid in sg_order:
        n = sg_nodes[nid]
        ntype = n["type"]
        if ntype in (SENTINEL_INPUT, SENTINEL_OUTPUT):
            continue
        if ntype in SKIP_NODES:
            continue
        if ntype in INLINE_NODES:
            model_nodes.add(nid)
            continue
        if ntype in MULTI_INPUT_NODES:
            model_nodes.add(nid)
            continue

        # Try to get the inner module
        safe_key = nid.replace('.', '_').replace('-', '_')
        inner_mod = None
        if hasattr(sg_module, 'inner_modules') and safe_key in sg_module.inner_modules:
            inner_mod = sg_module.inner_modules[safe_key]

        if inner_mod is None:
            continue

        code = _module_code(ntype, n.get("properties", {}), inner_mod)
        if code:
            model_nodes.add(nid)
            attr = _sanitize_id(nid)
            if code.startswith("__PRETRAINED_RESNET18__"):
                init_lines.append(f"        self.{attr} = self._build_resnet18()")
            else:
                init_lines.append(f"        self.{attr} = {code}")

    # Build forward body
    forward_body = _build_forward_body(sg_order, sg_nodes, sg_edges, {}, model_nodes)

    # Assemble class
    lines = []
    lines.append(f"class {class_name}(nn.Module):")
    lines.append(f'    """Subgraph block: {block_name}."""')
    lines.append("")
    lines.append("    def __init__(self):")
    lines.append("        super().__init__()")
    for il in init_lines:
        lines.append(il)
    lines.append("")
    lines.append("    def forward(self, x):")
    lines.append(forward_body)

    return '\n'.join(lines)


# ─── Main model class generation ─────────────────────────────────────────────

def _find_entry_concat_nodes(order: list[str], nodes: dict, edges: list) -> set[str]:
    """Find concat (or other multi-input) nodes whose ALL inputs come from SKIP_NODES.

    These are "entry points" where the training loop feeds pre-processed data
    into the model (e.g., diffusion concatenates noisy_images + timestep_channel
    before passing to the model). They should be excluded from the model class.
    """
    skip_node_ids = {nid for nid in order if nodes[nid]["type"] in SKIP_NODES}
    entry_nodes: set[str] = set()

    for nid in order:
        n = nodes[nid]
        if n["type"] not in MULTI_INPUT_NODES:
            continue
        # Check if ALL incoming edges come from skip nodes
        incoming = [e["source"]["nodeId"] for e in edges if e["target"]["nodeId"] == nid]
        if incoming and all(src in skip_node_ids for src in incoming):
            entry_nodes.add(nid)

    return entry_nodes


def _generate_model_class(
    order: list[str],
    nodes: dict,
    edges: list,
    modules: dict,
    training_mode: str,
    subgraph_classes: dict[str, str],  # node_id -> class_name
) -> str:
    """Generate the main Model class."""
    # Find entry-concat nodes that should be excluded from the model
    entry_concat_nodes = _find_entry_concat_nodes(order, nodes, edges)

    # Collect model nodes
    model_nodes: set[str] = set()
    init_lines: list[str] = []
    needs_resnet_helper = False

    for nid in order:
        n = nodes[nid]
        ntype = n["type"]

        if ntype in SKIP_NODES:
            continue
        if nid in entry_concat_nodes:
            continue
        if ntype in INLINE_NODES:
            model_nodes.add(nid)
            continue
        if ntype in MULTI_INPUT_NODES:
            model_nodes.add(nid)
            # Some multi-input nodes need a module in __init__ (e.g., MHA)
            # Others (Add, Concat, Reparameterize) are handled inline in forward
            module = modules.get(nid)
            if module is not None:
                code = _module_code(ntype, n.get("properties", {}), module)
                if code:
                    attr = _sanitize_id(nid)
                    init_lines.append(f"        self.{attr} = {code}")
            continue

        # Subgraph blocks
        if ntype == SUBGRAPH_TYPE:
            model_nodes.add(nid)
            class_name = subgraph_classes.get(nid, "Block")
            attr = _sanitize_id(nid)
            init_lines.append(f"        self.{attr} = {class_name}()")
            continue

        # Regular modules
        module = modules.get(nid)
        if module is None:
            continue

        code = _module_code(ntype, n.get("properties", {}), module)
        if code:
            model_nodes.add(nid)
            attr = _sanitize_id(nid)
            if code.startswith("__PRETRAINED_RESNET18__"):
                needs_resnet_helper = True
                mode = n.get("properties", {}).get("mode", "features")
                freeze = n.get("properties", {}).get("freeze", True)
                init_lines.append(f"        self.{attr} = self._build_resnet18(mode='{mode}', freeze={freeze})")
            else:
                init_lines.append(f"        self.{attr} = {code}")

    # Build forward body
    forward_body = _build_forward_body(order, nodes, edges, modules, model_nodes)

    # Assemble class
    lines = []
    lines.append("class Model(nn.Module):")
    lines.append('    """Neural network model generated by NodeTorch."""')
    lines.append("")
    lines.append("    def __init__(self):")
    lines.append("        super().__init__()")
    for il in init_lines:
        lines.append(il)
    lines.append("")

    if needs_resnet_helper:
        lines.append("    @staticmethod")
        lines.append("    def _build_resnet18(mode='features', freeze=True):")
        lines.append("        import torchvision.models as models")
        lines.append("        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)")
        lines.append("        if freeze:")
        lines.append("            for p in model.parameters():")
        lines.append("                p.requires_grad = False")
        lines.append("        if mode == 'features':")
        lines.append("            model.fc = nn.Identity()")
        lines.append("        return model")
        lines.append("")

    lines.append("    def forward(self, x):")
    lines.append(forward_body)

    return '\n'.join(lines)


# ─── Training loop templates ─────────────────────────────────────────────────

def _standard_classification_loop(opt_props: dict, opt_type: str, loss_node: dict | None) -> str:
    """Generate standard classification training loop."""
    lr = opt_props.get("lr", 0.001)
    epochs = opt_props.get("epochs", 10)
    scheduler_type = opt_props.get("scheduler", "")

    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append("    criterion = nn.CrossEntropyLoss()")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("        correct = 0")
    lines.append("        total = 0")
    lines.append("")
    lines.append("        for images, labels in train_loader:")
    lines.append("            images, labels = images.to(device), labels.to(device)")
    lines.append("")
    lines.append("            optimizer.zero_grad()")
    lines.append("            outputs = model(images)")
    lines.append("            loss = criterion(outputs, labels)")
    lines.append("            loss.backward()")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("            _, predicted = outputs.max(1)")
    lines.append("            total += labels.size(0)")
    lines.append("            correct += predicted.eq(labels).sum().item()")
    lines.append("")
    if scheduler_type:
        lines.append("        scheduler.step()")
    lines.append("        epoch_loss = running_loss / len(train_loader)")
    lines.append("        epoch_acc = 100.0 * correct / total")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}, Accuracy: {{epoch_acc:.1f}}%")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model")

    return '\n'.join(lines)


def _autoregressive_training_loop(opt_props: dict, opt_type: str) -> str:
    """Generate autoregressive LM training loop (per-token CrossEntropy, perplexity)."""
    import math as _math  # noqa: F401  (used in generated code)
    epochs = opt_props.get("epochs", 10)
    scheduler_type = opt_props.get("scheduler", "")
    grad_clip = opt_props.get("gradClip", 0)
    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    lines.append("import math")
    lines.append("")
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append("    criterion = nn.CrossEntropyLoss()")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("        n_batches = 0")
    lines.append("")
    lines.append("        for inputs, targets in train_loader:")
    lines.append("            inputs, targets = inputs.to(device), targets.to(device)")
    lines.append("")
    lines.append("            optimizer.zero_grad()")
    lines.append("            logits = model(inputs)  # [B, seq_len, vocab_size]")
    lines.append("            B, S, V = logits.shape")
    lines.append("            loss = criterion(logits.reshape(B * S, V), targets.reshape(B * S))")
    lines.append("            loss.backward()")
    if grad_clip and grad_clip > 0:
        lines.append(f"            torch.nn.utils.clip_grad_norm_(model.parameters(), {grad_clip})")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("            n_batches += 1")
    lines.append("")
    if scheduler_type:
        lines.append("        scheduler.step()")
    lines.append("        epoch_loss = running_loss / max(n_batches, 1)")
    lines.append("        ppl = math.exp(min(epoch_loss, 20))")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}, Perplexity: {{ppl:.2f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model")

    return '\n'.join(lines)


def _standard_reconstruction_loop(opt_props: dict, opt_type: str) -> str:
    """Generate MSE reconstruction training loop (autoencoder)."""
    lr = opt_props.get("lr", 0.001)
    epochs = opt_props.get("epochs", 10)
    scheduler_type = opt_props.get("scheduler", "")

    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append("    criterion = nn.MSELoss()")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("")
    lines.append("        for images, _ in train_loader:")
    lines.append("            images = images.to(device)")
    lines.append("")
    lines.append("            optimizer.zero_grad()")
    lines.append("            outputs = model(images)")
    lines.append("            loss = criterion(outputs, images)")
    lines.append("            loss.backward()")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("")
    if scheduler_type:
        lines.append("        scheduler.step()")
    lines.append("        epoch_loss = running_loss / len(train_loader)")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model")

    return '\n'.join(lines)


def _vae_training_loop(opt_props: dict, opt_type: str, loss_props: dict) -> str:
    """Generate VAE training loop."""
    epochs = opt_props.get("epochs", 15)
    beta = loss_props.get("beta", 1.0)
    scheduler_type = opt_props.get("scheduler", "")

    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    lines.append("def vae_loss_fn(reconstruction, original, mean, logvar, beta=1.0):")
    lines.append('    """VAE loss = reconstruction loss + beta * KL divergence."""')
    lines.append("    recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='mean')")
    lines.append("    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())")
    lines.append("    return recon_loss + beta * kl_loss")
    lines.append("")
    lines.append("")
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("")
    lines.append("        for images, _ in train_loader:")
    lines.append("            images = images.to(device)")
    lines.append("")
    lines.append("            optimizer.zero_grad()")
    lines.append("            reconstruction, mean, logvar = model(images)")
    lines.append(f"            loss = vae_loss_fn(reconstruction, images, mean, logvar, beta={beta})")
    lines.append("            loss.backward()")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("")
    if scheduler_type:
        lines.append("        scheduler.step()")
    lines.append("        epoch_loss = running_loss / len(train_loader)")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model")

    return '\n'.join(lines)


def _gan_training_loop(graph_data: dict, modules: dict) -> str:
    """Generate GAN training loop with Generator and Discriminator."""
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}

    # Find noise node for latent dim
    latent_dim = 100
    batch_size = 64
    for n in nodes.values():
        if n["type"] == GAN_NOISE_TYPE:
            latent_dim = n.get("properties", {}).get("latentDim", 100)
            batch_size = n.get("properties", {}).get("batchSize", 64)

    # Find optimizer(s) for lr, epochs
    opt_nodes = [n for n in nodes.values() if n["type"] in OPTIMIZER_NODES]
    # Use first optimizer's settings
    opt_props = opt_nodes[0].get("properties", {}) if opt_nodes else {"lr": 0.0002, "epochs": 100}
    epochs = opt_props.get("epochs", 100)
    lr_g = opt_props.get("lr", 0.0002)
    # Second optimizer (if exists) for discriminator
    lr_d = opt_nodes[1].get("properties", {}).get("lr", 0.0001) if len(opt_nodes) > 1 else lr_g
    beta1 = opt_props.get("beta1", 0.5)
    beta2 = opt_props.get("beta2", 0.999)
    label_smoothing = 0.1
    for n in nodes.values():
        if n["type"] == "ml.loss.gan":
            label_smoothing = n.get("properties", {}).get("labelSmoothing", 0.1)

    lines = []
    lines.append("def train():")
    lines.append("    generator = Generator().to(device)")
    lines.append("    discriminator = Discriminator().to(device)")
    lines.append("")
    lines.append(f"    optimizer_g = torch.optim.Adam(generator.parameters(), lr={lr_g}, betas=({beta1}, {beta2}))")
    lines.append(f"    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr={lr_d}, betas=({beta1}, {beta2}))")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        generator.train()")
    lines.append("        discriminator.train()")
    lines.append("        d_loss_total = 0.0")
    lines.append("        g_loss_total = 0.0")
    lines.append("")
    lines.append("        for real_images, _ in train_loader:")
    lines.append("            real_images = real_images.to(device)")
    lines.append(f"            batch_size = real_images.size(0)")
    lines.append("")
    lines.append("            # --- Train Discriminator ---")
    lines.append("            optimizer_d.zero_grad()")
    lines.append(f"            noise = torch.randn(batch_size, {latent_dim}, device=device)")
    lines.append("            fake_images = generator(noise).detach()")
    lines.append("")
    lines.append("            real_scores = discriminator(real_images)")
    lines.append("            fake_scores = discriminator(fake_images)")
    lines.append("")
    lines.append(f"            real_labels = torch.ones_like(real_scores) * {1.0 - label_smoothing}")
    lines.append("            fake_labels = torch.zeros_like(fake_scores)")
    lines.append("            d_loss_real = nn.functional.binary_cross_entropy_with_logits(real_scores, real_labels)")
    lines.append("            d_loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_scores, fake_labels)")
    lines.append("            d_loss = d_loss_real + d_loss_fake")
    lines.append("            d_loss.backward()")
    lines.append("            optimizer_d.step()")
    lines.append("")
    lines.append("            # --- Train Generator ---")
    lines.append("            optimizer_g.zero_grad()")
    lines.append(f"            noise = torch.randn(batch_size, {latent_dim}, device=device)")
    lines.append("            fake_images = generator(noise)")
    lines.append("            fake_scores = discriminator(fake_images)")
    lines.append("            g_loss = nn.functional.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))")
    lines.append("            g_loss.backward()")
    lines.append("            optimizer_g.step()")
    lines.append("")
    lines.append("            d_loss_total += d_loss.item()")
    lines.append("            g_loss_total += g_loss.item()")
    lines.append("")
    lines.append("        n_batches = len(train_loader)")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — D Loss: {{d_loss_total/n_batches:.4f}}, G Loss: {{g_loss_total/n_batches:.4f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return generator, discriminator")

    return '\n'.join(lines)


def _diffusion_training_loop(graph_data: dict, modules: dict) -> str:
    """Generate diffusion training loop with noise scheduler."""
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}

    # Find scheduler properties
    num_timesteps = 100
    beta_start = 0.0001
    beta_end = 0.02
    schedule_type = "linear"
    for n in nodes.values():
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            props = n.get("properties", {})
            num_timesteps = props.get("numTimesteps", 100)
            beta_start = props.get("betaStart", 0.0001)
            beta_end = props.get("betaEnd", 0.02)
            schedule_type = props.get("scheduleType", "linear")

    # Optimizer props
    opt_props = _get_optimizer_props(nodes)
    opt_type = _get_optimizer_type(nodes)
    epochs = opt_props.get("epochs", 50)
    scheduler_type = opt_props.get("scheduler", "")

    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    # Noise scheduler class
    lines.append("class NoiseScheduler:")
    lines.append('    """DDPM noise scheduler for diffusion training."""')
    lines.append("")
    lines.append(f"    def __init__(self, num_timesteps={num_timesteps}, beta_start={beta_start}, beta_end={beta_end}):")
    if schedule_type == "cosine":
        lines.append("        # Cosine schedule")
        lines.append("        steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps")
        lines.append("        alpha_bar = torch.cos((steps + 0.008) / 1.008 * 3.14159265 / 2) ** 2")
        lines.append("        alpha_bar = alpha_bar / alpha_bar[0]")
        lines.append("        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])")
        lines.append("        self.betas = torch.clamp(betas, 0.0001, 0.999).float()")
    else:
        lines.append("        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)")
    lines.append("        self.alphas = 1.0 - self.betas")
    lines.append("        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)")
    lines.append("        self.num_timesteps = num_timesteps")
    lines.append("")
    lines.append("    def add_noise(self, x, noise, t):")
    lines.append('        """Add noise at timestep t: x_t = sqrt(alpha_bar_t) * x + sqrt(1-alpha_bar_t) * noise."""')
    lines.append("        sqrt_alpha = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1).to(x.device)")
    lines.append("        sqrt_one_minus = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1).to(x.device)")
    lines.append("        return sqrt_alpha * x + sqrt_one_minus * noise")
    lines.append("")
    lines.append("    def sample_step(self, model, x_t, t):")
    lines.append('        """One denoising step: predict noise, remove it."""')
    lines.append("        beta = self.betas[t].to(x_t.device)")
    lines.append("        alpha = self.alphas[t].to(x_t.device)")
    lines.append("        alpha_bar = self.alpha_cumprod[t].to(x_t.device)")
    lines.append("")
    lines.append("        # Predict noise")
    lines.append("        t_channel = torch.full((x_t.size(0), 1, x_t.size(2), x_t.size(3)), t / self.num_timesteps, device=x_t.device)")
    lines.append("        model_input = torch.cat([x_t, t_channel], dim=1)")
    lines.append("        predicted_noise = model(model_input)")
    lines.append("")
    lines.append("        # Compute x_{t-1}")
    lines.append("        x_prev = (1 / alpha.sqrt()) * (x_t - (beta / (1 - alpha_bar).sqrt()) * predicted_noise)")
    lines.append("        if t > 0:")
    lines.append("            noise = torch.randn_like(x_t)")
    lines.append("            x_prev = x_prev + beta.sqrt() * noise")
    lines.append("        return x_prev")
    lines.append("")
    lines.append("")
    # Training function
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append(f"    scheduler = NoiseScheduler(num_timesteps={num_timesteps})")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    lr_scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("    criterion = nn.MSELoss()")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("")
    lines.append("        for images, _ in train_loader:")
    lines.append("            images = images.to(device)")
    lines.append("            batch_size = images.size(0)")
    lines.append("")
    lines.append("            # Sample random timesteps")
    lines.append(f"            t = torch.randint(0, {num_timesteps}, (batch_size,))")
    lines.append("            noise = torch.randn_like(images)")
    lines.append("            noisy_images = scheduler.add_noise(images, noise, t)")
    lines.append("")
    lines.append("            # Concatenate timestep channel")
    lines.append(f"            t_normalized = t.float() / {num_timesteps}")
    lines.append("            t_channel = t_normalized.view(-1, 1, 1, 1).expand(-1, 1, images.size(2), images.size(3)).to(device)")
    lines.append("            model_input = torch.cat([noisy_images, t_channel], dim=1)")
    lines.append("")
    lines.append("            # Predict and compute loss")
    lines.append("            optimizer.zero_grad()")
    lines.append("            predicted_noise = model(model_input)")
    lines.append("            loss = criterion(predicted_noise, noise)")
    lines.append("            loss.backward()")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("")
    if scheduler_type:
        lines.append("        lr_scheduler.step()")
    lines.append("        epoch_loss = running_loss / len(train_loader)")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model, scheduler")
    lines.append("")
    lines.append("")
    # Sampling function
    lines.append("@torch.no_grad()")
    lines.append("def sample(model, scheduler, num_samples=16):")
    lines.append('    """Generate images by denoising from pure noise."""')
    lines.append("    model.eval()")
    lines.append(f"    x = torch.randn(num_samples, {_get_image_channels_for_diffusion(nodes)}, {_get_image_size_for_diffusion(nodes)}, {_get_image_size_for_diffusion(nodes)}, device=device)")
    lines.append(f"    for t in reversed(range({num_timesteps})):")
    lines.append("        x = scheduler.sample_step(model, x, t)")
    lines.append("    return x.clamp(-1, 1)")

    return '\n'.join(lines)


def _get_image_channels_for_diffusion(nodes: dict) -> int:
    """Get image channels from the data node."""
    for n in nodes.values():
        if n["type"] in DATASET_CODE:
            return DATASET_CODE[n["type"]]["channels"]
    return 1


def _get_image_size_for_diffusion(nodes: dict) -> int:
    """Get image size from the data node."""
    for n in nodes.values():
        if n["type"] in DATASET_CODE:
            return DATASET_CODE[n["type"]]["image_size"]
    return 28


# ─── Optimizer code generation ────────────────────────────────────────────────

def _optimizer_code(opt_type: str, props: dict) -> str:
    """Generate optimizer constructor call."""
    lr = props.get("lr", 0.001)
    if opt_type == "ml.optimizers.sgd":
        momentum = props.get("momentum", 0.9)
        wd = props.get("weightDecay", 0)
        code = f"torch.optim.SGD(model.parameters(), lr={lr}, momentum={momentum}"
        if wd:
            code += f", weight_decay={wd}"
        return code + ")"
    elif opt_type == "ml.optimizers.adamw":
        wd = props.get("weightDecay", 0.01)
        return f"torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay={wd})"
    else:  # adam
        beta1 = props.get("beta1", 0.9)
        beta2 = props.get("beta2", 0.999)
        if beta1 != 0.9 or beta2 != 0.999:
            return f"torch.optim.Adam(model.parameters(), lr={lr}, betas=({beta1}, {beta2}))"
        return f"torch.optim.Adam(model.parameters(), lr={lr})"


def _scheduler_code(scheduler_type: str, epochs: int) -> str:
    """Generate LR scheduler code."""
    if scheduler_type == "cosine":
        return f"torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={epochs})"
    elif scheduler_type == "step":
        return f"torch.optim.lr_scheduler.StepLR(optimizer, step_size={max(1, epochs // 3)}, gamma=0.1)"
    return ""


# ─── Dataset loading code generation ─────────────────────────────────────────

def _text_dataset_code(hf_name: str, props: dict, num_classes: int) -> str:
    """Generate HuggingFace text dataset loading with a simple word-level vocab."""
    batch_size = props.get("batchSize", 32)
    max_len = props.get("maxLen", 256 if hf_name == "imdb" else 128)
    vocab_size = props.get("vocabSize", 10000)

    lines = [
        "# --- Dataset (text: word-level vocab built from training split) ---",
        "# Requires `datasets` package: pip install datasets",
        "import re",
        "from collections import Counter",
        "from datasets import load_dataset",
        "",
        f"_raw_train = load_dataset('{hf_name}', split='train')",
        f"_raw_test = load_dataset('{hf_name}', split='test')",
        "",
        "def _tokenize(text):",
        "    return re.findall(r'[a-z]+', text.lower())",
        "",
        "# Build vocab from first 5k training samples (matches NodeTorch backend)",
        "_counter = Counter()",
        "for _t in _raw_train['text'][:5000]:",
        "    _counter.update(_tokenize(_t))",
        "vocab = {'<pad>': 0, '<unk>': 1}",
        f"for _w, _ in _counter.most_common({vocab_size} - 2):",
        "    vocab[_w] = len(vocab)",
        "UNK = vocab['<unk>']",
        "",
        "class TextDataset(torch.utils.data.Dataset):",
        "    def __init__(self, raw):",
        "        self.raw = raw",
        "    def __len__(self):",
        "        return len(self.raw)",
        "    def __getitem__(self, idx):",
        "        text = self.raw[idx]['text']",
        "        label = self.raw[idx]['label']",
        "        tokens = _tokenize(text)",
        f"        ids = [vocab.get(t, UNK) for t in tokens[:{max_len}]]",
        f"        ids = ids + [0] * ({max_len} - len(ids))",
        "        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)",
        "",
        "train_dataset = TextDataset(_raw_train)",
        "test_dataset = TextDataset(_raw_test)",
        f"train_loader = torch.utils.data.DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, num_workers=0)",
        f"test_loader = torch.utils.data.DataLoader(test_dataset, batch_size={batch_size}, shuffle=False, num_workers=0)",
    ]
    return '\n'.join(lines)


def _shakespeare_dataset_code(props: dict) -> str:
    """Generate TinyShakespeare character-level dataset code."""
    batch_size = props.get("batchSize", 64)
    seq_len = props.get("seqLen", 128)

    lines = [
        "# --- Dataset (Tiny Shakespeare: character-level language modeling) ---",
        "import os, urllib.request",
        "",
        "_SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'",
        "_path = './data/tiny_shakespeare.txt'",
        "os.makedirs('./data', exist_ok=True)",
        "if not os.path.exists(_path):",
        "    urllib.request.urlretrieve(_SHAKESPEARE_URL, _path)",
        "with open(_path, 'r', encoding='utf-8') as _f:",
        "    _text = _f.read()",
        "",
        "_chars = sorted(set(_text))",
        "char2idx = {c: i for i, c in enumerate(_chars)}",
        "idx2char = {i: c for c, i in char2idx.items()}",
        "VOCAB_SIZE = len(char2idx)",
        "_data = torch.tensor([char2idx[c] for c in _text], dtype=torch.long)",
        "",
        "class TinyShakespeareDataset(torch.utils.data.Dataset):",
        f"    SEQ_LEN = {seq_len}",
        "    def __len__(self):",
        "        return (len(_data) - 1) // self.SEQ_LEN",
        "    def __getitem__(self, idx):",
        "        start = idx * self.SEQ_LEN",
        "        chunk = _data[start : start + self.SEQ_LEN + 1]",
        "        if len(chunk) < self.SEQ_LEN + 1:",
        "            chunk = torch.cat([chunk, torch.zeros(self.SEQ_LEN + 1 - len(chunk), dtype=torch.long)])",
        "        return chunk[:-1], chunk[1:]",
        "",
        "train_dataset = TinyShakespeareDataset()",
        f"train_loader = torch.utils.data.DataLoader(train_dataset, batch_size={batch_size}, shuffle=True, num_workers=0)",
        "# No separate test set for char-level LM — use perplexity on held-out text if needed",
        "test_loader = train_loader",
    ]
    return '\n'.join(lines)


def _generate_dataset_code(data_node: dict) -> str:
    """Generate dataset loading code."""
    dtype = data_node["type"]
    props = data_node.get("properties", {})
    batch_size = props.get("batchSize", 32)

    # Text datasets (HuggingFace) — sentiment / topic classification
    if dtype == "data.imdb":
        return _text_dataset_code("imdb", props, num_classes=2)
    if dtype == "data.ag_news":
        return _text_dataset_code("ag_news", props, num_classes=4)
    # Character-level language model
    if dtype == "data.tiny_shakespeare":
        return _shakespeare_dataset_code(props)

    info = DATASET_CODE.get(dtype)
    if not info:
        return f"# TODO: Add dataset loading for {dtype}\ntrain_loader = None"

    # Check augmentation flags
    aug_hflip = props.get("augHFlip", False)
    aug_crop = props.get("augRandomCrop", False)
    aug_jitter = props.get("augColorJitter", False)

    lines = []
    lines.append("# --- Dataset ---")
    lines.append("")

    # Build transform list
    transform_items = []
    if aug_crop:
        size = info["image_size"]
        transform_items.append(f"    transforms.RandomCrop({size}, padding=4),")
    if aug_hflip:
        transform_items.append("    transforms.RandomHorizontalFlip(),")
    if aug_jitter:
        transform_items.append("    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),")
    transform_items.append("    transforms.ToTensor(),")
    transform_items.append(f"    transforms.Normalize({info['norm_mean']}, {info['norm_std']}),")

    lines.append("transform = transforms.Compose([")
    for item in transform_items:
        lines.append(item)
    lines.append("])")
    lines.append("")
    lines.append(f"train_dataset = {info['class']}(")
    lines.append(f"    root='./data', train=True, download=True, transform=transform,")
    lines.append(")")
    lines.append(f"train_loader = torch.utils.data.DataLoader(")
    lines.append(f"    train_dataset, batch_size={batch_size}, shuffle=True, num_workers=2,")
    lines.append(")")
    lines.append("")
    # Test dataset (no augmentation)
    lines.append("test_transform = transforms.Compose([")
    lines.append("    transforms.ToTensor(),")
    lines.append(f"    transforms.Normalize({info['norm_mean']}, {info['norm_std']}),")
    lines.append("])")
    lines.append(f"test_dataset = {info['class']}(")
    lines.append(f"    root='./data', train=False, download=True, transform=test_transform,")
    lines.append(")")
    lines.append(f"test_loader = torch.utils.data.DataLoader(")
    lines.append(f"    test_dataset, batch_size={batch_size}, shuffle=False, num_workers=2,")
    lines.append(")")

    return '\n'.join(lines)


# ─── Main export function ─────────────────────────────────────────────────────

def export_to_python(graph_data: dict) -> str:
    """Generate a standalone Python file from a NodeTorch graph.

    Args:
        graph_data: Serialized graph in NodeTorch format { version, graph: { nodes, edges } }

    Returns:
        Complete, runnable Python file as a string.
    """
    graph = graph_data["graph"]
    nodes_list = graph["nodes"]
    edges = graph["edges"]
    graph_name = graph.get("name", "Model")

    nodes = {n["id"]: n for n in nodes_list}
    order = topological_sort(nodes, edges)

    # Build modules to resolve actual dimensions
    try:
        modules = build_modules(graph_data)
    except Exception:
        modules = {}

    # Detect training paradigm
    training_mode = detect_training_mode(nodes)

    # Find key nodes
    data_node = _get_data_node(nodes)
    loss_node = _get_loss_node(nodes)
    opt_props = _get_optimizer_props(nodes)
    opt_type = _get_optimizer_type(nodes)

    # ─── Generate subgraph classes ────────────────────────────────────────────
    subgraph_classes: dict[str, str] = {}  # node_id -> class_name
    subgraph_code_blocks: list[str] = []

    for nid in order:
        n = nodes[nid]
        if n["type"] == SUBGRAPH_TYPE and n.get("subgraph"):
            code = _generate_subgraph_class(n, modules, modules)
            block_name = n.get("properties", {}).get("blockName", "Block")
            class_name = re.sub(r'[^a-zA-Z0-9]', '', block_name)
            if not class_name:
                class_name = "Block"
            subgraph_classes[nid] = class_name
            subgraph_code_blocks.append(code)

    # ─── Generate model class ─────────────────────────────────────────────────

    if training_mode == "gan":
        # GAN: Generator and Discriminator are subgraph blocks, no main Model class
        model_code = ""
    else:
        model_code = _generate_model_class(order, nodes, edges, modules, training_mode, subgraph_classes)

    # ─── Generate training loop ───────────────────────────────────────────────

    if training_mode == "gan":
        training_code = _gan_training_loop(graph_data, modules)
    elif training_mode == "diffusion":
        training_code = _diffusion_training_loop(graph_data, modules)
    elif training_mode == "autoregressive":
        training_code = _autoregressive_training_loop(opt_props, opt_type)
    elif loss_node and loss_node["type"] == "ml.loss.vae":
        loss_props = loss_node.get("properties", {})
        training_code = _vae_training_loop(opt_props, opt_type, loss_props)
    elif loss_node and loss_node["type"] == "ml.loss.mse":
        # Check if it's actually diffusion (MSE used for noise prediction)
        training_code = _standard_reconstruction_loop(opt_props, opt_type)
    else:
        training_code = _standard_classification_loop(opt_props, opt_type, loss_node)

    # ─── Assemble the complete file ───────────────────────────────────────────

    sections: list[str] = []

    # Header
    sections.append(f'"""')
    sections.append(f'{graph_name}')
    sections.append(f'Generated by NodeTorch — https://github.com/wilsonzeng/nodetorch')
    sections.append(f'')
    sections.append(f'A standalone, runnable training script.')
    sections.append(f'Run: python {graph_name.replace(" ", "_").lower()}.py')
    sections.append(f'"""')
    sections.append("")

    # Imports
    sections.append("import torch")
    sections.append("import torch.nn as nn")
    sections.append("import torchvision")
    sections.append("import torchvision.transforms as transforms")
    sections.append("")

    # Device
    sections.append("# --- Device selection ---")
    sections.append("")
    sections.append("if torch.cuda.is_available():")
    sections.append("    device = torch.device('cuda')")
    sections.append("elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():")
    sections.append("    device = torch.device('mps')")
    sections.append("else:")
    sections.append("    device = torch.device('cpu')")
    sections.append("print(f'Using device: {device}')")
    sections.append("")

    # Dataset
    if data_node:
        sections.append(_generate_dataset_code(data_node))
        sections.append("")

    # Subgraph classes
    if subgraph_code_blocks:
        sections.append("")
        sections.append("# --- Network blocks ---")
        sections.append("")
        for block_code in subgraph_code_blocks:
            sections.append(block_code)
            sections.append("")

    # Model class (not for GAN — subgraph blocks serve as models)
    if model_code:
        sections.append("")
        sections.append("# --- Model ---")
        sections.append("")
        sections.append(model_code)
        sections.append("")

    # Training loop
    sections.append("")
    sections.append("# --- Training ---")
    sections.append("")
    sections.append(training_code)
    sections.append("")

    # Main
    sections.append("")
    sections.append("# --- Entry point ---")
    sections.append("")
    sections.append('if __name__ == "__main__":')
    if training_mode == "gan":
        sections.append("    generator, discriminator = train()")
        sections.append("    # Generate samples from trained GAN")
        latent_dim = 100
        for n in nodes.values():
            if n["type"] == GAN_NOISE_TYPE:
                latent_dim = n.get("properties", {}).get("latentDim", 100)
        sections.append(f"    noise = torch.randn(16, {latent_dim}, device=device)")
        sections.append("    with torch.no_grad():")
        sections.append("        generated = generator(noise)")
        sections.append("    print(f'Generated images shape: {generated.shape}')")
    elif training_mode == "diffusion":
        sections.append("    model, scheduler = train()")
        sections.append("    # Generate samples by denoising")
        sections.append("    samples = sample(model, scheduler, num_samples=16)")
        sections.append("    print(f'Generated samples shape: {samples.shape}')")
    elif loss_node and loss_node["type"] == "ml.loss.vae":
        sections.append("    model = train()")
        sections.append("    # Generate by sampling from latent space")
        sections.append("    with torch.no_grad():")
        sections.append("        z = torch.randn(16, 32, device=device)  # sample from N(0,1)")
        sections.append("        # Pass z through decoder (requires adapting the model)")
        sections.append("        print('VAE training complete. See model for encoding/decoding.')")
    elif loss_node and loss_node["type"] == "ml.loss.mse":
        # Reconstruction (autoencoder): report MSE, no accuracy
        sections.append("    model = train()")
        sections.append("    # Evaluate reconstruction MSE on test set")
        sections.append("    model.eval()")
        sections.append("    total_mse = 0.0")
        sections.append("    n_batches = 0")
        sections.append("    criterion_eval = nn.MSELoss()")
        sections.append("    with torch.no_grad():")
        sections.append("        for images, _ in test_loader:")
        sections.append("            images = images.to(device)")
        sections.append("            outputs = model(images)")
        sections.append("            total_mse += criterion_eval(outputs, images).item()")
        sections.append("            n_batches += 1")
        sections.append("    print(f'Test reconstruction MSE: {total_mse / max(n_batches, 1):.4f}')")
    elif data_node and data_node["type"] == "data.tiny_shakespeare":
        # Autoregressive language model: report perplexity, skip accuracy
        sections.append("    import math")
        sections.append("    model = train()")
        sections.append("    # Evaluate perplexity on held-out sequences")
        sections.append("    model.eval()")
        sections.append("    total_loss = 0.0")
        sections.append("    n_tokens = 0")
        sections.append("    criterion_eval = nn.CrossEntropyLoss(reduction='sum')")
        sections.append("    with torch.no_grad():")
        sections.append("        for inputs, targets in test_loader:")
        sections.append("            inputs, targets = inputs.to(device), targets.to(device)")
        sections.append("            logits = model(inputs)")
        sections.append("            B, S, V = logits.shape")
        sections.append("            total_loss += criterion_eval(logits.reshape(B*S, V), targets.reshape(B*S)).item()")
        sections.append("            n_tokens += B * S")
        sections.append("    avg = total_loss / max(n_tokens, 1)")
        sections.append("    print(f'Test perplexity: {math.exp(min(avg, 20)):.2f}')")
    else:
        sections.append("    model = train()")
        sections.append("    # Evaluate on test set")
        sections.append("    model.eval()")
        sections.append("    correct = 0")
        sections.append("    total = 0")
        sections.append("    with torch.no_grad():")
        sections.append("        for images, labels in test_loader:")
        sections.append("            images, labels = images.to(device), labels.to(device)")
        sections.append("            outputs = model(images)")
        sections.append("            _, predicted = outputs.max(1)")
        sections.append("            total += labels.size(0)")
        sections.append("            correct += predicted.eq(labels).sum().item()")
        sections.append("    print(f'Test accuracy: {100.0 * correct / total:.1f}%')")
    sections.append("")

    return '\n'.join(sections)
