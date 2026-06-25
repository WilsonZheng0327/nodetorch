"""Inference, held-out test-set evaluation, and tracked-sample probing.

``infer_graph`` runs a single sample through the stored trained modules;
``evaluate_test_set`` scores the trained model on the held-out split. The
``_pick_tracked_samples`` / ``_probe_tracked_samples`` / ``_collect_misclassifications``
helpers support the training dashboard and are imported by ``training/``.
"""

import torch
import torch.nn as nn

from dataprep.data_loaders import DATA_LOADERS, DENORMALIZERS
from engine.graph_builder.constants import (
    LOSS_NODES, OPTIMIZER_NODES, MULTI_INPUT_NODES, ALL_LOSS_NODES,
    GAN_NOISE_TYPE, DIFFUSION_SCHEDULER_TYPE, SUBGRAPH_TYPE,
)
from engine.graph_builder._state import (
    get_device, has_trained_model, get_trained_modules, ensure_trained_model, _last_run,
)
from engine.graph_builder.build import topological_sort, gather_inputs
from engine.graph_builder.stats import tensor_info, _safe_float


def evaluate_test_set(graph_data: dict) -> dict:
    """Evaluate trained model on the held-out test set.

    Returns test loss, test accuracy, per-class accuracy, and sample count.
    Only works for classification models (2D predictions).
    """
    # Lazy-load the on-disk snapshot if the in-memory store is empty (e.g. the
    # backend restarted since training). ensure_trained_model() returns whether a
    # model is available afterward.
    if not ensure_trained_model():
        return {"error": "No trained model — train first"}

    # Diffusion and GAN models don't support standard test evaluation
    graph = graph_data["graph"]
    for n in graph["nodes"]:
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            return {"error": "Diffusion models don't use test set evaluation — use Step Through > Denoise to generate samples"}
        if n["type"] == GAN_NOISE_TYPE:
            return {"error": "GAN models don't use test set evaluation — check generated samples in the training dashboard"}
    from dataprep.data_loaders import LM_DATASET_TYPES
    for n in graph["nodes"]:
        if n["type"] in LM_DATASET_TYPES:
            return {"error": "Language models don't use test set evaluation — use Step Through > Generate to produce text samples"}

    trained_modules = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    # Find data, loss, and prediction nodes
    data_node = None
    loss_node_id = None
    pred_node_id = None
    for nid in order:
        n = nodes[nid]
        if n["type"] in DATA_LOADERS:
            data_node = n
        if n["type"] in ALL_LOSS_NODES:
            loss_node_id = nid
            for edge in edges:
                if edge["target"]["nodeId"] == nid and edge["target"]["portId"] == "predictions":
                    pred_node_id = edge["source"]["nodeId"]

    if not data_node:
        return {"error": "No data node in graph"}

    # Load test dataset
    from dataprep.data_loaders import TEST_DATASETS, CLASS_NAMES
    dataset_type = data_node["type"]
    test_builder = TEST_DATASETS.get(dataset_type)
    if not test_builder:
        return {"error": f"No test set available for {dataset_type}"}

    data_props = data_node.get("properties", {})
    import inspect
    if inspect.signature(test_builder).parameters:
        test_dataset = test_builder(**{
            k: data_props[k] for k in inspect.signature(test_builder).parameters
            if k in data_props
        })
    else:
        test_dataset = test_builder()

    batch_size = data_props.get("batchSize", 32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dev = get_device()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    all_preds: list[int] = []
    all_labels: list[int] = []

    for mod in trained_modules.values():
        if isinstance(mod, nn.Module):
            mod.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(dev), labels.to(dev)

            batch_results: dict[str, dict] = {}
            data_nid = data_node["id"]
            batch_results[data_nid] = {"out": images, "labels": labels}

            for nid in order:
                if nid == data_nid:
                    continue
                n = nodes[nid]
                ntype = n["type"]
                if ntype in OPTIMIZER_NODES:
                    continue
                mod = trained_modules.get(nid)
                if mod is None:
                    continue

                inputs = gather_inputs(nid, edges, batch_results)

                if ntype in LOSS_NODES:
                    if "predictions" in inputs and "labels" in inputs:
                        loss = mod(inputs["predictions"], inputs["labels"])
                        batch_results[nid] = {"out": loss}
                        total_loss += loss.item()
                    continue

                if ntype == SUBGRAPH_TYPE:
                    sg_out = mod(**inputs)
                    first_key = next(iter(sg_out), None)
                    if first_key:
                        batch_results[nid] = {"out": sg_out[first_key]}
                    continue

                if ntype in MULTI_INPUT_NODES:
                    batch_results[nid] = {"out": mod(**{k: v for k, v in inputs.items()})}
                    continue

                if "in" in inputs:
                    raw = mod(inputs["in"])
                    if isinstance(raw, dict):
                        batch_results[nid] = raw
                    else:
                        batch_results[nid] = {"out": raw}

            # Accuracy + confusion data (classification only)
            if pred_node_id and pred_node_id in batch_results:
                preds = batch_results[pred_node_id].get("out")
                if preds is not None and preds.dim() == 2:
                    predicted = preds.argmax(dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    all_preds.extend(predicted.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                    for cls in labels.unique().tolist():
                        cls = int(cls)
                        mask = labels == cls
                        per_class_total[cls] = per_class_total.get(cls, 0) + mask.sum().item()
                        per_class_correct[cls] = per_class_correct.get(cls, 0) + (predicted[mask] == cls).sum().item()

    # Set modules back to train mode
    for mod in trained_modules.values():
        if isinstance(mod, nn.Module):
            mod.train()

    n_batches = len(test_loader)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    accuracy = correct / total if total > 0 else 0

    class_names = CLASS_NAMES.get(dataset_type, [])
    per_class = []
    for cls in sorted(per_class_total.keys()):
        acc = per_class_correct.get(cls, 0) / per_class_total[cls] if per_class_total.get(cls, 0) > 0 else 0
        name = class_names[cls] if cls < len(class_names) else str(cls)
        per_class.append({"cls": cls, "name": name, "accuracy": _safe_float(acc), "count": per_class_total[cls]})

    # Build confusion matrix
    confusion = None
    if all_preds and all_labels:
        n_classes = max(max(all_preds), max(all_labels)) + 1
        matrix = [[0] * n_classes for _ in range(n_classes)]
        for p, l in zip(all_preds, all_labels):
            if 0 <= p < n_classes and 0 <= l < n_classes:
                matrix[l][p] += 1
        confusion = {
            "data": matrix,
            "size": n_classes,
            "classNames": class_names,
        }

    return {
        "testLoss": _safe_float(avg_loss),
        "testAccuracy": _safe_float(accuracy),
        "testSamples": total,
        "perClassAccuracy": per_class,
        "confusionMatrix": confusion,
    }


def infer_graph(graph_data: dict) -> dict:
    """
    Run inference using trained weights.
    Loads a single sample, runs through stored trained modules,
    returns per-node results + prediction.
    """
    # Lazy-load the on-disk snapshot if the in-memory store is empty (e.g. the
    # backend restarted since training). ensure_trained_model() returns whether a
    # model is available afterward.
    if not ensure_trained_model():
        return {"error": "No trained model — train first"}

    # Diffusion and GAN models don't support standard inference
    graph = graph_data["graph"]
    for n in graph["nodes"]:
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            return {"error": "Diffusion models generate images via denoising — use Step Through > Denoise tab"}
        if n["type"] == GAN_NOISE_TYPE:
            return {"error": "GAN inference generates images from noise — use Step Through > Denoise or check training dashboard for generated samples"}
    from dataprep.data_loaders import LM_DATASET_TYPES
    for n in graph["nodes"]:
        if n["type"] in LM_DATASET_TYPES:
            return {"error": "Language models generate text autoregressively — use Step Through > Generate tab"}

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
                    tensors = {k: (v.to(get_device()) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
                    results[node_id] = tensors
                    outputs = {k: tensor_info(v) for k, v in tensors.items() if isinstance(v, torch.Tensor)}
                    first_tensor = next(v for v in tensors.values() if isinstance(v, torch.Tensor))

                    # Include the actual label for display (scalar labels only, not sequences)
                    lbl_tensor = tensors.get("labels")
                    label = int(lbl_tensor[0]) if lbl_tensor is not None and isinstance(lbl_tensor, torch.Tensor) and lbl_tensor.dim() == 1 else None

                    meta: dict = {
                        "outputShape": list(first_tensor.shape),
                        "actualLabel": label,
                    }

                    # Image datasets: send raw pixel data for preview
                    if "out" in tensors and isinstance(tensors["out"], torch.Tensor) and tensors["out"].dim() == 4:
                        img = tensors["out"][0].detach().cpu()
                        C = img.shape[0]
                        denorm = DENORMALIZERS.get(node_type)
                        if denorm:
                            img = denorm(img)
                        img = (img.clamp(0, 1) * 255).byte()
                        if C == 1:
                            meta["imagePixels"] = img[0].tolist()
                            meta["imageChannels"] = 1
                        else:
                            meta["imagePixels"] = img.permute(1, 2, 0).tolist()
                            meta["imageChannels"] = C

                    # Text datasets: send raw sample text for preview
                    if "_texts" in tensors:
                        meta["sampleText"] = tensors["_texts"][0][:500]

                    node_results[node_id] = {
                        "outputs": outputs,
                        "metadata": meta,
                    }
                except Exception as e:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": str(e)},
                    }
                continue

            # Optimizer/loss: skip during inference
            if node_type in OPTIMIZER_NODES or node_type in ALL_LOSS_NODES:
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
                raw = module(inputs["in"])

                # Handle multi-output (LSTM/GRU)
                if isinstance(raw, dict):
                    results[node_id] = raw
                    output = next(iter(raw.values()))
                else:
                    results[node_id] = {"out": raw}
                    output = raw

                meta: dict = {
                    "outputShape": list(output.shape),
                    "paramCount": sum(p.numel() for p in module.parameters()),
                }

                is_final = any(
                    e["source"]["nodeId"] == node_id
                    and nodes.get(e["target"]["nodeId"], {}).get("type") in LOSS_NODES
                    for e in edges
                )
                if is_final and output.dim() == 2:
                    # Classification: show predicted class + probabilities
                    probs = torch.softmax(output, dim=1)[0]
                    predicted_class = int(probs.argmax())
                    confidence = float(probs[predicted_class])
                    prediction = {
                        "predictedClass": predicted_class,
                        "confidence": _safe_float(confidence),
                        "probabilities": [_safe_float(float(p)) for p in probs],
                    }
                    meta["prediction"] = prediction
                elif is_final and output.dim() == 4:
                    # Reconstruction (autoencoder): show output as image
                    img = output[0].detach().cpu()  # [C, H, W]
                    img = (img.clamp(0, 1) * 255).byte()
                    C = img.shape[0]
                    if C == 1:
                        meta["imagePixels"] = img[0].tolist()
                        meta["imageChannels"] = 1
                    else:
                        meta["imagePixels"] = img.permute(1, 2, 0).tolist()
                        meta["imageChannels"] = C
                    meta["reconstruction"] = True

                out_info = {k: tensor_info(v) for k, v in raw.items()} if isinstance(raw, dict) else {"out": tensor_info(output)}
                node_results[node_id] = {
                    "outputs": out_info,
                    "metadata": meta,
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }

    # Cache for layer detail queries
    _last_run.clear()
    _last_run["modules"] = trained_modules
    _last_run["results"] = results
    _last_run["nodes"] = nodes
    _last_run["edges"] = edges

    return {
        "nodeResults": node_results,
        "prediction": prediction,
    }


def _pick_tracked_samples(dataset, dataset_type: str, n: int = 4) -> list[dict]:
    """Pick N fixed samples from the dataset to track across epochs.

    Returns a list of dicts, each with:
      - idx: index in the dataset
      - label: class label (int)
      - image: pixel data for display (if image dataset)
      - text: raw text (if text dataset)
      - input: the raw input tensor (for forwarding through the model)
    """
    import random
    from dataprep.data_loaders import DENORMALIZERS

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    samples = []

    for idx in indices:
        item = dataset[idx]
        # Most datasets return (input_tensor, label_tensor)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            inp, lbl = item
        else:
            continue

        sample: dict = {
            "idx": idx,
            "label": int(lbl) if isinstance(lbl, int) or (isinstance(lbl, torch.Tensor) and lbl.dim() == 0) else None,
            "input": inp if isinstance(inp, torch.Tensor) else torch.tensor(inp),
        }

        # Image preview (4D-capable datasets)
        if isinstance(inp, torch.Tensor) and inp.dim() == 3:
            img = inp.detach().cpu()
            denorm = DENORMALIZERS.get(dataset_type)
            if denorm:
                img = denorm(img)
            img = (img.clamp(0, 1) * 255).byte()
            C = img.shape[0]
            if C == 1:
                sample["imagePixels"] = img[0].tolist()
                sample["imageChannels"] = 1
            else:
                sample["imagePixels"] = img.permute(1, 2, 0).tolist()
                sample["imageChannels"] = C

        samples.append(sample)

    return samples


def _probe_tracked_samples(
    tracked: list[dict],
    modules: dict,
    nodes: dict,
    edges: list,
    order: list[str],
    dataset_type: str,
) -> list[dict]:
    """Run tracked samples through the current model and record predictions.

    Returns a list matching the tracked samples, each with:
      - idx, label (from tracked)
      - imagePixels, imageChannels (from tracked, only on first epoch for efficiency)
      - probabilities: softmax output (if classification, i.e. 2D output)
      - predictedClass, confidence
      - loss: per-sample loss value (if available)
      - output: summary of the final layer output (for non-classification models)
    """
    if not tracked:
        return []

    dev = get_device()
    data_nid = None
    loss_nid = None
    for nid in order:
        n = nodes[nid]
        if n["type"] in DATA_LOADERS:
            data_nid = nid
        if n["type"] in ALL_LOSS_NODES:
            loss_nid = nid

    if not data_nid:
        return []

    # Find the node feeding predictions to loss (final layer)
    pred_nid = None
    if loss_nid:
        for edge in edges:
            if (edge["target"]["nodeId"] == loss_nid
                    and edge["target"]["portId"] == "predictions"):
                pred_nid = edge["source"]["nodeId"]
                break

    results = []
    for s in tracked:
        inp = s["input"].unsqueeze(0).to(dev)  # add batch dim
        label_tensor = torch.tensor([s["label"]], dtype=torch.long).to(dev) if s["label"] is not None else None

        # Forward pass through the graph
        batch_results: dict[str, dict] = {}
        batch_results[data_nid] = {"out": inp, "labels": label_tensor}

        with torch.no_grad():
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
                mod.eval()

                inputs = gather_inputs(nid, edges, batch_results)

                if ntype in LOSS_NODES:
                    if "predictions" in inputs and "labels" in inputs:
                        try:
                            loss_val = mod(inputs["predictions"], inputs["labels"])
                            batch_results[nid] = {"out": loss_val}
                        except Exception:
                            pass
                    continue

                if ntype == SUBGRAPH_TYPE:
                    try:
                        sg_out = mod(**inputs)
                        first_key = next(iter(sg_out), None)
                        if first_key:
                            batch_results[nid] = {"out": sg_out[first_key]}
                    except Exception:
                        pass
                    continue

                if ntype in MULTI_INPUT_NODES:
                    try:
                        batch_results[nid] = {"out": mod(**{k: v for k, v in inputs.items()})}
                    except Exception:
                        pass
                    continue

                if "in" in inputs:
                    try:
                        raw = mod(inputs["in"])
                        if isinstance(raw, dict):
                            batch_results[nid] = raw
                        else:
                            batch_results[nid] = {"out": raw}
                    except Exception:
                        pass

        # Set modules back to train mode
        for mod in modules.values():
            if isinstance(mod, nn.Module):
                mod.train()

        # Build probe result
        probe: dict = {
            "idx": s["idx"],
            "label": s["label"],
        }

        # Only send image/text on first call (frontend caches it)
        if "imagePixels" in s:
            probe["imagePixels"] = s["imagePixels"]
            probe["imageChannels"] = s.get("imageChannels", 1)

        # Extract prediction from the final layer
        if pred_nid and pred_nid in batch_results:
            pred_out = batch_results[pred_nid].get("out")
            if pred_out is not None and isinstance(pred_out, torch.Tensor):
                if pred_out.dim() == 2:
                    # Classification: softmax probabilities
                    probs = torch.softmax(pred_out, dim=1)[0]
                    probe["probabilities"] = [_safe_float(float(p)) for p in probs.tolist()]
                    pred_class = int(probs.argmax())
                    probe["predictedClass"] = pred_class
                    probe["confidence"] = _safe_float(float(probs[pred_class]))
                else:
                    # Non-classification (autoencoder etc): just report output norm
                    probe["outputNorm"] = _safe_float(float(pred_out.detach().float().norm()))

        # Per-sample loss
        if loss_nid and loss_nid in batch_results:
            loss_out = batch_results[loss_nid].get("out")
            if loss_out is not None and isinstance(loss_out, torch.Tensor):
                probe["loss"] = _safe_float(float(loss_out.item()))

        results.append(probe)

    return results


def _collect_misclassifications(
    images: torch.Tensor,
    predicted: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    dataset_type: str,
    samples: list,
    counts: dict,
    max_per_pair: int = 4,
    max_total: int = 50,
) -> None:
    """Collect a cap of misclassified samples from the current batch.

    Each sample includes displayable pixels (denormalized), predicted/actual labels,
    and softmax confidence for the predicted class. Caps storage so response stays small.
    """
    if len(samples) >= max_total or images.dim() != 4:
        return

    wrong = predicted != labels
    if not wrong.any():
        return

    denorm = DENORMALIZERS.get(dataset_type)
    probs_batch = torch.softmax(logits, dim=1)

    for i in range(labels.size(0)):
        if not bool(wrong[i]):
            continue
        actual = int(labels[i])
        pred = int(predicted[i])
        key = (actual, pred)
        if counts.get(key, 0) >= max_per_pair:
            continue
        if len(samples) >= max_total:
            break
        counts[key] = counts.get(key, 0) + 1

        img = images[i].detach().cpu()
        if denorm:
            img = denorm(img)
        img = (img.clamp(0, 1) * 255).byte()
        C = img.shape[0]
        if C == 1:
            pixels = img[0].tolist()
        else:
            pixels = img.permute(1, 2, 0).tolist()

        samples.append({
            "actual": actual,
            "predicted": pred,
            "confidence": _safe_float(float(probs_batch[i, pred])),
            "imagePixels": pixels,
            "imageChannels": int(C),
        })
