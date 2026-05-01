"""Step-through — run a forward pass and record intermediate activations at each node.

Requires a trained model. Supports label-filtered and index-based sample selection.
GAN graphs get special handling: runs generator and discriminator on both real and fake.
"""

from __future__ import annotations
import copy
import torch

from graph_builder import (
    build_and_run_graph,
    gather_inputs,
    topological_sort,
    has_trained_model,
    get_trained_modules,
    get_device,
    _safe_float,
    LOSS_NODES,
    ALL_LOSS_NODES,
    OPTIMIZER_NODES,
    SUBGRAPH_TYPE,
    SENTINEL_INPUT,
    SENTINEL_OUTPUT,
    GAN_NOISE_TYPE,
    DIFFUSION_SCHEDULER_TYPE,
    SubGraphModule,
)
from forward_utils import execute_node
from data_loaders import DATA_LOADERS, DENORMALIZERS, CLASS_NAMES, load_sample_by_label
from node_viz import get_forward_viz


def run_step_through(
    graph_data: dict,
    filter_label: int | None = None,
    sample_idx: int | None = None,
) -> dict:
    if not has_trained_model():
        raise RuntimeError("Train the model first — step-through requires trained weights")

    graph_data = copy.deepcopy(graph_data)
    for n in graph_data["graph"]["nodes"]:
        if n["type"] in DATA_LOADERS:
            n["properties"] = {**n.get("properties", {}), "batchSize": 1}

    # Detect graph type
    node_types = {n["type"] for n in graph_data["graph"]["nodes"]}
    is_gan = GAN_NOISE_TYPE in node_types
    is_diffusion = DIFFUSION_SCHEDULER_TYPE in node_types

    try:
        if is_gan:
            modules, results, nodes, edges, used_idx = _forward_gan(
                graph_data, filter_label=filter_label, sample_idx=sample_idx,
            )
        elif is_diffusion:
            modules, results, nodes, edges, used_idx = _forward_diffusion(
                graph_data, filter_label=filter_label, sample_idx=sample_idx,
            )
        else:
            modules, results, nodes, edges, used_idx = _forward_with_trained(
                graph_data, filter_label=filter_label, sample_idx=sample_idx,
            )
    except Exception as e:
        raise RuntimeError(f"Step-through failed: {e}")

    order = topological_sort(nodes, edges)
    stages: list[dict] = []
    sample_info = _extract_sample_info(nodes, results)

    for node_id in order:
        node = nodes[node_id]
        node_type = node["type"]

        if node_type in OPTIMIZER_NODES:
            continue

        if node_type == SUBGRAPH_TYPE:
            sg_module = modules.get(node_id)
            if isinstance(sg_module, SubGraphModule):
                block_name = node.get("properties", {}).get("blockName") or node_id
                inner_stages = _build_subgraph_stages(
                    sg_module=sg_module, block_name=block_name, depth=1, parent_path=[node_id],
                )
                stages.extend(inner_stages)
                continue

        stage = _build_stage(
            node=node, node_id=node_id, path=[node_id], depth=0,
            edges=edges, results=results, nodes=nodes, module=modules.get(node_id),
        )
        if stage:
            stages.append(stage)

    return {
        "stages": stages,
        "sample": sample_info,
        "sampleIdx": used_idx,
    }


# --- Standard forward pass ---

def _forward_with_trained(graph_data, filter_label=None, sample_idx=None):
    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    results: dict[str, dict] = {}
    dev = get_device()
    used_idx = -1

    with torch.no_grad():
        for node_id in order:
            node = nodes[node_id]
            node_type = node["type"]
            props = node.get("properties", {})

            if node_type in DATA_LOADERS:
                tensors, used_idx = load_sample_by_label(
                    node_type, props, filter_label=filter_label, sample_idx=sample_idx,
                )
                tensors = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
                results[node_id] = tensors
                continue

            if node_type in OPTIMIZER_NODES:
                continue
            if node_type in ALL_LOSS_NODES:
                continue

            module = trained.get(node_id)
            if module is None:
                raise RuntimeError(f"Trained model missing module for node {node_id}")

            inputs = gather_inputs(node_id, edges, results)
            try:
                out = execute_node(node_type, module, inputs)
            except Exception:
                continue
            if out is not None:
                results[node_id] = out

    return trained, results, nodes, edges, used_idx


# --- Diffusion-aware forward pass ---

def _forward_diffusion(graph_data, filter_label=None, sample_idx=None):
    """Diffusion step-through: add noise at a random timestep, run model to predict noise."""
    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)
    dev = get_device()
    used_idx = -1

    # Find scheduler and data nodes
    scheduler_nid = None
    data_nid = None
    for nid, n in nodes.items():
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            scheduler_nid = nid
        elif n["type"] in DATA_LOADERS:
            data_nid = nid

    results: dict[str, dict] = {}

    with torch.no_grad():
        # 1. Load real image
        if data_nid:
            props = nodes[data_nid].get("properties", {})
            tensors, used_idx = load_sample_by_label(
                nodes[data_nid]["type"], props, filter_label=filter_label, sample_idx=sample_idx,
            )
            tensors = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
            results[data_nid] = tensors

        # 2. Noise scheduler: add noise at a random timestep
        if scheduler_nid:
            scheduler_module = trained.get(scheduler_nid)
            images_input = gather_inputs(scheduler_nid, edges, results)
            images = images_input.get("images")

            if scheduler_module and images is not None:
                # Pick a mid-range timestep for visualization
                num_timesteps = getattr(scheduler_module, 'num_timesteps', 100)
                import random as _rand
                t_val = _rand.randint(0, num_timesteps - 1)
                t = torch.tensor([t_val], device=dev, dtype=torch.long)

                noise = torch.randn_like(images)
                noisy_images = scheduler_module.add_noise(images, noise, t)

                # Timestep channel
                t_normalized = (t.float() / num_timesteps).view(-1, 1, 1, 1)
                t_channel = t_normalized.expand(images.shape[0], 1, images.shape[2], images.shape[3])

                results[scheduler_nid] = {
                    "out": noisy_images,
                    "noise": noise,
                    "timestep": t_channel,
                }

        # 3. Run remaining nodes
        for node_id in order:
            if node_id in results:
                continue
            node = nodes[node_id]
            node_type = node["type"]
            if node_type in OPTIMIZER_NODES or node_type in ALL_LOSS_NODES:
                continue
            if node_type in DATA_LOADERS or node_type == DIFFUSION_SCHEDULER_TYPE:
                continue

            module = trained.get(node_id)
            if module is None:
                continue

            inputs = gather_inputs(node_id, edges, results)
            try:
                out = execute_node(node_type, module, inputs)
            except Exception:
                continue
            if out is not None:
                results[node_id] = out

    return trained, results, nodes, edges, used_idx


# --- GAN-aware forward pass ---

def _forward_gan(graph_data, filter_label=None, sample_idx=None):
    """GAN step-through: runs generator, then discriminator on real AND fake.

    1. Load real image from data node
    2. Generate noise and run generator → fake image
    3. Run discriminator on real image → real_scores
    4. Run discriminator on fake image → fake_scores
    5. Compute GAN loss from both scores
    """
    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)
    dev = get_device()
    used_idx = -1

    # Identify components
    noise_nid = None
    data_nid = None
    loss_nid = None
    gen_nid = None
    disc_nid = None

    for nid, n in nodes.items():
        if n["type"] == GAN_NOISE_TYPE:
            noise_nid = nid
        elif n["type"] in DATA_LOADERS:
            data_nid = nid
        elif n["type"] == "ml.loss.gan":
            loss_nid = nid

    # Find generator and discriminator subgraph blocks
    # Generator: downstream of noise. Discriminator: feeds into GAN loss.
    downstream: dict[str, list[str]] = {nid: [] for nid in nodes}
    for edge in edges:
        downstream[edge["source"]["nodeId"]].append(edge["target"]["nodeId"])

    if noise_nid:
        noise_reachable = set()
        queue = [noise_nid]
        while queue:
            cur = queue.pop(0)
            for child in downstream.get(cur, []):
                if child not in noise_reachable:
                    noise_reachable.add(child)
                    queue.append(child)

        for nid in order:
            if nid in noise_reachable and nodes[nid]["type"] == SUBGRAPH_TYPE:
                if gen_nid is None:
                    gen_nid = nid
                elif disc_nid is None:
                    disc_nid = nid

    if disc_nid is None:
        for nid in order:
            if nodes[nid]["type"] == SUBGRAPH_TYPE and nid != gen_nid:
                disc_nid = nid
                break

    results: dict[str, dict] = {}

    with torch.no_grad():
        # 1. Load real image
        if data_nid:
            props = nodes[data_nid].get("properties", {})
            tensors, used_idx = load_sample_by_label(
                nodes[data_nid]["type"], props, filter_label=filter_label, sample_idx=sample_idx,
            )
            tensors = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
            results[data_nid] = tensors
            real_images = tensors.get("out")

        # 2. Generate noise and run through generator
        if noise_nid:
            latent_dim = nodes[noise_nid].get("properties", {}).get("latentDim", 100)
            noise = torch.randn(1, latent_dim, device=dev)
            results[noise_nid] = {"out": noise}

        # Run all nodes in order, skipping loss/optimizer
        for node_id in order:
            if node_id in results:
                continue
            node = nodes[node_id]
            node_type = node["type"]
            if node_type in OPTIMIZER_NODES or node_type in ALL_LOSS_NODES:
                continue
            if node_type in DATA_LOADERS or node_type == GAN_NOISE_TYPE:
                continue

            module = trained.get(node_id)
            if module is None:
                continue

            inputs = gather_inputs(node_id, edges, results)
            try:
                out = execute_node(node_type, module, inputs)
            except Exception:
                continue
            if out is not None:
                results[node_id] = out

        # At this point, the generator has produced fake images,
        # and the discriminator has run on whatever was connected to it.
        # For the GAN loss, we need real_scores and fake_scores.
        # The discriminator may have only run on fake (since gen→disc path exists).
        # We need to also run it on real.

        # Find what the disc produced (fake scores) and re-run on real
        if disc_nid and data_nid:
            disc_module = trained.get(disc_nid)
            real_images = results.get(data_nid, {}).get("out")

            if disc_module and isinstance(disc_module, SubGraphModule) and real_images is not None:
                # Save the fake-path results as they are
                fake_disc_result = results.get(disc_nid)
                fake_scores = fake_disc_result.get("out") if fake_disc_result else None

                # Run discriminator on real images
                disc_module.eval()
                try:
                    real_out = disc_module(**{"in": real_images})
                    first_key = next(iter(real_out), None)
                    real_scores = real_out[first_key] if first_key else None
                except Exception:
                    real_scores = None

                # Now compute GAN loss if we have both scores
                if loss_nid and fake_scores is not None and real_scores is not None:
                    loss_module = trained.get(loss_nid)
                    if loss_module:
                        try:
                            loss_val = loss_module(real_scores, fake_scores)
                            results[loss_nid] = {
                                "out": loss_val,
                                "real_scores": real_scores,
                                "fake_scores": fake_scores,
                            }
                        except Exception:
                            pass

    return trained, results, nodes, edges, used_idx


# --- Subgraph recursion ---

def _build_subgraph_stages(*, sg_module, block_name, depth, parent_path):
    inner_nodes = sg_module.inner_nodes
    inner_edges = sg_module.inner_edges
    inner_order = sg_module.inner_order
    inner_results = getattr(sg_module, '_last_results', {})

    stages = []
    for inner_nid in inner_order:
        inner_node = inner_nodes[inner_nid]
        inner_type = inner_node["type"]
        if inner_type in (SENTINEL_INPUT, SENTINEL_OUTPUT) or inner_type in OPTIMIZER_NODES:
            continue

        safe_key = sg_module._key_map.get(inner_nid)
        inner_mod = sg_module.inner_modules[safe_key] if safe_key and safe_key in sg_module.inner_modules else None
        if isinstance(inner_mod, SubGraphModule):
            nested_name = inner_node.get("properties", {}).get("blockName") or inner_nid
            stages.extend(_build_subgraph_stages(
                sg_module=inner_mod, block_name=nested_name,
                depth=depth + 1, parent_path=parent_path + [inner_nid],
            ))
            continue

        stage = _build_stage(
            node=inner_node, node_id=inner_nid, path=parent_path + [inner_nid],
            depth=depth, edges=inner_edges, results=inner_results, nodes=inner_nodes, module=inner_mod,
        )
        if stage:
            stage["blockName"] = block_name
            stages.append(stage)
    return stages


# --- Sample extraction ---

def _extract_sample_info(nodes, results):
    for node_id, node in nodes.items():
        if node["type"] in DATA_LOADERS:
            tensors = results.get(node_id, {})
            out = tensors.get("out")
            labels = tensors.get("labels")
            dataset_type = node["type"]
            info: dict = {
                "datasetType": dataset_type,
                "actualLabel": int(labels[0].item()) if labels is not None and isinstance(labels, torch.Tensor) and labels.dim() == 1 else None,
            }
            if dataset_type in CLASS_NAMES:
                info["classNames"] = CLASS_NAMES[dataset_type]
            if out is not None and isinstance(out, torch.Tensor) and out.dim() == 4:
                info.update(_tensor_to_preview_image(out[0], dataset_type))
            elif out is not None and isinstance(out, torch.Tensor) and out.dim() == 2:
                info["tokenIds"] = out[0].tolist()[:64]
                if "_texts" in tensors:
                    info["sampleText"] = tensors["_texts"][0][:500]
            return info
    return {}


def _tensor_to_preview_image(img, dataset_type):
    denorm = DENORMALIZERS.get(dataset_type)
    if denorm:
        img = denorm(img)
    img = (img.clamp(0, 1) * 255).byte()
    C = img.shape[0]
    if C == 1:
        return {"imagePixels": img[0].tolist(), "imageChannels": 1}
    return {"imagePixels": img.permute(1, 2, 0).tolist(), "imageChannels": C}


# --- Stage building ---

def _build_stage(*, node, node_id, path, depth, edges, results, nodes, module=None):
    node_type = node["type"]
    out_dict = results.get(node_id, {})
    output = out_dict.get("out") if isinstance(out_dict, dict) else None
    if output is None or not isinstance(output, torch.Tensor):
        if node_type not in ALL_LOSS_NODES:
            return None

    inputs = gather_inputs(node_id, edges, results)

    # For GAN loss, also inject real/fake scores from results if available
    if node_type == "ml.loss.gan" and node_id in results:
        r = results[node_id]
        if "real_scores" in r:
            inputs["real_scores"] = r["real_scores"]
        if "fake_scores" in r:
            inputs["fake_scores"] = r["fake_scores"]

    input_tensor = None
    for v in inputs.values():
        if isinstance(v, torch.Tensor):
            input_tensor = v
            break

    stage = {
        "stageId": "/".join(path), "path": path, "nodeId": node_id,
        "nodeType": node_type, "displayName": _friendly_name(node_type),
        "depth": depth,
        "inputShape": list(input_tensor.shape) if input_tensor is not None else None,
        "outputShape": list(output.shape) if output is not None else ([1] if node_type in ALL_LOSS_NODES else None),
    }

    try:
        viz_result = get_forward_viz(node_type, module, input_tensor, output, inputs, out_dict)
    except Exception as e:
        import logging
        logging.getLogger("nodetorch").error(
            f"Viz failed for {node_type} (node {node_id}): {e} | "
            f"input={input_tensor.shape if input_tensor is not None and hasattr(input_tensor, 'shape') else None} "
            f"output={output.shape if output is not None and hasattr(output, 'shape') else None}"
        )
        viz_result = {}
    if viz_result.get("transformation"):
        stage["transformation"] = viz_result["transformation"]
    if viz_result.get("insight"):
        stage["insight"] = viz_result["insight"]
    return stage


def _friendly_name(node_type):
    last = node_type.split(".")[-1]
    if "_" in last:
        return "".join(p.capitalize() for p in last.split("_"))
    return last[0].upper() + last[1:]
