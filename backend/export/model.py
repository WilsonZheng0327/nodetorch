"""Model class generation: forward body, subgraph classes, the nn.Module."""

import re
from engine.graph_builder import (
    MULTI_INPUT_NODES, SENTINEL_INPUT, SENTINEL_OUTPUT, SUBGRAPH_TYPE, topological_sort,
)
from export.helpers import _sanitize_id
from export.layers import INLINE_NODES, MULTI_OUTPUT_NODES, SKIP_NODES, _inline_code, _module_code

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
            causal = n.get("properties", {}).get("causalMask", False)
            if causal:
                lines.append(f"_S = {q_var}.shape[1]")
                lines.append(f"_mask = torch.triu(torch.ones(_S, _S, device={q_var}.device, dtype=torch.bool), diagonal=1)")
                call = f"{attr_name}({q_var}, {k_var}, {v_var}, attn_mask=_mask, need_weights=False)"
            else:
                call = f"{attr_name}({q_var}, {k_var}, {v_var}, need_weights=False)"
            if nid in needs_var:
                vname = var_names[nid]
                lines.append(f"{vname}, _ = {call}")
                last_var = vname
            else:
                lines.append(f"x, _ = {call}")
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
    needs_pos_encoding = False
    needs_tokenizer = False

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
            elif code.startswith("__POSITIONAL_ENCODING__"):
                needs_pos_encoding = True
                args = code[len("__POSITIONAL_ENCODING__"):]
                init_lines.append(f"        self.{attr} = PositionalEncoding{args}")
            elif code.startswith("__TOKENIZER__"):
                needs_tokenizer = True
                args = code[len("__TOKENIZER__"):]
                init_lines.append(f"        self.{attr} = Tokenizer{args}")
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


