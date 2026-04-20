"""Standard training loop — single forward pass, single loss, single optimizer.

Handles classification (with accuracy/confusion matrix) and reconstruction
(autoencoders/VAEs — loss only, no accuracy). This is the default training
paradigm used by most models in NodeTorch.
"""

from __future__ import annotations
import time
import torch
from tqdm import tqdm

from graph_builder import (
    get_device,
    _safe_float,
    ALL_LOSS_NODES,
    _pick_tracked_samples,
    _probe_tracked_samples,
    _collect_misclassifications,
)
from data_loaders import CLASS_NAMES
from forward_utils import run_forward_pass

from .base import (
    TrainingContext,
    TrainingResult,
    build_optimizer,
    build_scheduler,
    init_weight_norms,
    compute_batch_accuracy,
    run_validation_pass,
    collect_node_snapshots,
    build_gradient_flow,
    build_epoch_result,
    run_final_forward,
)


def standard_train(ctx: TrainingContext) -> TrainingResult:
    """Run the standard training loop: forward → loss → backward → update."""

    # Collect all trainable parameters
    all_params = list(p for m in ctx.modules.values() for p in m.parameters())

    # Create optimizer and scheduler
    primary_opt = ctx.optimizer_nodes[0]
    props = primary_opt["properties"]
    optimizer = build_optimizer(primary_opt, all_params)
    scheduler = build_scheduler(optimizer, props.get("scheduler", "none"), ctx.epochs)

    # Primary loss node
    loss_node_id = ctx.loss_node_ids[0]

    # State
    prev_weight_norms = init_weight_norms(ctx.modules)
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    early_stopped = False
    epoch_results = []
    total_batches = len(ctx.train_loader)
    dev = get_device()

    for epoch in range(ctx.epochs):
        if ctx.cancel_event and ctx.cancel_event.is_set():
            break

        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        per_class_correct: dict[int, int] = {}
        per_class_total: dict[int, int] = {}
        confusion_preds: list[int] = []
        confusion_labels: list[int] = []
        misclass_samples: list[dict] = []
        misclass_counts: dict[tuple, int] = {}
        last_batch_results: dict = {}

        batch_report_interval = max(1, total_batches // 20)
        for batch_idx, (images, labels) in enumerate(tqdm(
            ctx.train_loader, desc=f"Epoch {epoch+1}/{ctx.epochs}", leave=False, ncols=80,
        )):
            if ctx.cancel_event and ctx.cancel_event.is_set():
                break
            images, labels = images.to(dev), labels.to(dev)

            if ctx.on_batch and batch_idx % batch_report_interval == 0:
                ctx.on_batch({"epoch": epoch + 1, "totalEpochs": ctx.epochs,
                              "batch": batch_idx + 1, "totalBatches": total_batches})

            optimizer.zero_grad()

            # Forward pass
            data_inputs = {ctx.data_node_id: {"out": images, "labels": labels}}
            batch_results = run_forward_pass(ctx.modules, ctx.nodes, ctx.edges, ctx.order, data_inputs)

            # Backward
            loss_tensor = batch_results.get(loss_node_id, {}).get("out")
            if loss_tensor is not None:
                loss_tensor.backward()
                if ctx.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, ctx.grad_clip_norm)
                optimizer.step()
                total_loss += loss_tensor.item()

                # Accuracy (classification only)
                c, t, pcc, pct = compute_batch_accuracy(batch_results, loss_node_id, ctx.edges, labels)
                correct += c
                total += t
                for cls in pct:
                    per_class_total[cls] = per_class_total.get(cls, 0) + pct[cls]
                    per_class_correct[cls] = per_class_correct.get(cls, 0) + pcc.get(cls, 0)

                # Confusion matrix + misclassifications
                for edge in ctx.edges:
                    if edge["target"]["nodeId"] == loss_node_id and edge["target"]["portId"] == "predictions":
                        pred_nid = edge["source"]["nodeId"]
                        preds = batch_results.get(pred_nid, {}).get("out")
                        if preds is not None and preds.dim() == 2:
                            predicted = preds.argmax(dim=1)
                            confusion_preds.extend(predicted.cpu().tolist())
                            confusion_labels.extend(labels.cpu().tolist())
                            _collect_misclassifications(
                                images, predicted, labels, preds,
                                ctx.dataset_type, misclass_samples, misclass_counts,
                            )
                        break

            last_batch_results = batch_results

        # Epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(ctx.train_loader)
        accuracy = correct / total if total > 0 else 0.0

        # Validation
        val_loss, val_accuracy = run_validation_pass(ctx, ctx.modules, loss_node_id)

        # Early stopping
        if ctx.early_stop_patience > 0 and val_loss is not None:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= ctx.early_stop_patience:
                    early_stopped = True

        # Snapshots
        node_snapshots = collect_node_snapshots(
            ctx.modules, last_batch_results, ctx.nodes, ctx.order, prev_weight_norms,
        )
        gradient_flow = build_gradient_flow(node_snapshots, ctx.nodes, ctx.order)

        # Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # Tracked samples
        tracked_probes = _probe_tracked_samples(
            ctx.tracked_samples, ctx.modules, ctx.nodes, ctx.edges, ctx.order, ctx.dataset_type,
        )

        # Per-class accuracy
        per_class_accuracy = sorted([
            {"cls": cls, "accuracy": _safe_float(
                per_class_correct.get(cls, 0) / per_class_total[cls]) if per_class_total.get(cls, 0) > 0 else 0.0}
            for cls in per_class_total.keys()
        ], key=lambda x: x["accuracy"]) if per_class_total else []

        # Build and send epoch result
        epoch_result = build_epoch_result(
            epoch, ctx, avg_loss, accuracy, val_loss, val_accuracy,
            current_lr, epoch_time, total_batches, total,
            gradient_flow, per_class_accuracy, node_snapshots, tracked_probes,
        )
        epoch_results.append(epoch_result)

        if ctx.on_epoch:
            ctx.on_epoch(epoch_result)

        if early_stopped:
            print(f"Early stopping at epoch {epoch + 1} (patience={ctx.early_stop_patience})")
            break

    # Final forward pass
    final_results, node_results = run_final_forward(ctx, ctx.modules)

    # Confusion matrix
    confusion_data = None
    if confusion_preds and confusion_labels:
        n_classes = max(max(confusion_preds), max(confusion_labels)) + 1
        matrix = [[0] * n_classes for _ in range(n_classes)]
        for p, l in zip(confusion_preds, confusion_labels):
            if 0 <= p < n_classes and 0 <= l < n_classes:
                matrix[l][p] += 1
        confusion_data = {
            "data": matrix,
            "size": n_classes,
            "classNames": CLASS_NAMES.get(ctx.dataset_type, []),
        }

    return TrainingResult(
        epoch_results=epoch_results,
        modules=ctx.modules,
        node_results=node_results,
        final_results=final_results,
        confusion_data=confusion_data,
        misclassifications=misclass_samples if misclass_samples else None,
        training_mode="standard",
    )
