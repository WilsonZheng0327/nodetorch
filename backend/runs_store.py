"""Training run history — save completed runs to disk and load them back.

Each run is a JSON file in ./runs/ keyed by timestamp. Stores:
  - id, timestamp, duration
  - optimizer config (lr, epochs, scheduler, seed, etc.)
  - dataset type + batch size
  - per-epoch results (loss, valLoss, accuracy, valAccuracy, learningRate, time)
  - final metrics (bestValAccuracy, finalLoss)
  - a summary of the graph (node count, param count)

Graph snapshots are intentionally NOT stored per run to keep files small —
but we save the optimizer + data node config so users know how each run differed.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

RUNS_DIR = Path("./storage/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def save_run(record: dict) -> str:
    """Save a run record. Returns the run ID."""
    run_id = record.get("id") or f"run-{int(time.time() * 1000)}"
    record["id"] = run_id
    path = RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2))
    return run_id


def list_runs() -> list[dict]:
    """List all runs (summary only — no per-epoch data)."""
    runs = []
    for f in RUNS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            runs.append({
                "id": data.get("id"),
                "timestamp": data.get("timestamp"),
                "datasetType": data.get("datasetType"),
                "epochs": data.get("epochs"),
                "learningRate": data.get("learningRate"),
                "optimizer": data.get("optimizer"),
                "scheduler": data.get("scheduler"),
                "finalLoss": data.get("finalLoss"),
                "finalAccuracy": data.get("finalAccuracy"),
                "bestValAccuracy": data.get("bestValAccuracy"),
                "duration": data.get("duration"),
                "totalParams": data.get("totalParams"),
                "nodeCount": data.get("nodeCount"),
            })
        except Exception:
            continue
    # Newest first
    runs.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return runs


def load_run(run_id: str) -> dict | None:
    """Load full run record including per-epoch history."""
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def delete_run(run_id: str) -> bool:
    path = RUNS_DIR / f"{run_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def build_run_record(
    *,
    graph_data: dict,
    epoch_results: list[dict],
    optimizer_props: dict,
    data_node: dict,
    duration_seconds: float,
    module_param_count: int,
) -> dict:
    """Build a run record dict from training state. Callers pass the data to save."""
    best_val_acc = None
    final_val_acc = None
    final_acc = None
    final_loss = None
    for e in epoch_results:
        if e.get("valAccuracy") is not None:
            final_val_acc = e["valAccuracy"]
            if best_val_acc is None or e["valAccuracy"] > best_val_acc:
                best_val_acc = e["valAccuracy"]
        if e.get("accuracy") is not None:
            final_acc = e["accuracy"]
        if e.get("loss") is not None:
            final_loss = e["loss"]

    # Trim heavy fields from epoch history (keep the essentials for charting)
    trimmed_epochs = []
    for e in epoch_results:
        trimmed_epochs.append({
            "epoch": e.get("epoch"),
            "loss": e.get("loss"),
            "accuracy": e.get("accuracy"),
            "valLoss": e.get("valLoss"),
            "valAccuracy": e.get("valAccuracy"),
            "learningRate": e.get("learningRate"),
            "time": e.get("time"),
        })

    opt_type = optimizer_props.get("__type__", "unknown")

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasetType": data_node["type"],
        "epochs": optimizer_props.get("epochs"),
        "learningRate": optimizer_props.get("lr"),
        "batchSize": data_node.get("properties", {}).get("batchSize"),
        "optimizer": opt_type,
        "scheduler": optimizer_props.get("scheduler", "none"),
        "seed": optimizer_props.get("seed"),
        "valSplit": optimizer_props.get("valSplit"),
        "finalLoss": final_loss,
        "finalAccuracy": final_acc,
        "finalValAccuracy": final_val_acc,
        "bestValAccuracy": best_val_acc,
        "duration": round(duration_seconds, 1),
        "totalParams": module_param_count,
        "nodeCount": len(graph_data.get("graph", {}).get("nodes", [])),
        "epochHistory": trimmed_epochs,
    }
