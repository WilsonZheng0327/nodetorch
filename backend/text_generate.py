"""Text generation from a trained autoregressive model.

Iterative decode loop: encode prompt → forward → sample next token → append → repeat.
Supports temperature scaling and top-k sampling.
"""

from __future__ import annotations
import torch

from graph_builder import (
    get_device,
    has_trained_model,
    get_trained_modules,
    topological_sort,
    gather_inputs,
    OPTIMIZER_NODES,
    ALL_LOSS_NODES,
)
from forward_utils import run_forward_pass
from data_loaders import get_shakespeare_vocab, LM_DATASET_TYPES


def _find_vocab_size(nodes: dict) -> int | None:
    """Determine vocab size from dataset node type."""
    for n in nodes.values():
        if n.get("type") in LM_DATASET_TYPES:
            char2idx, _ = get_shakespeare_vocab()
            return len(char2idx)
    return None


def generate_text(
    graph_data: dict,
    prompt: str = "",
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 0,
) -> dict:
    """Generate text autoregressively from a trained model.

    Args:
        graph_data: serialized graph JSON
        prompt: seed text (encoded to token IDs)
        max_tokens: how many tokens to generate
        temperature: sampling temperature (lower = more deterministic)
        top_k: if > 0, only sample from top-k most likely tokens

    Returns:
        { prompt, generated, fullText, tokens: [{char, prob}] }
    """
    if not has_trained_model():
        return {"error": "No trained model — train first"}

    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    # Get vocab
    char2idx, idx2char = get_shakespeare_vocab()
    vocab_size = len(char2idx)

    # Find data node
    data_nid = None
    for nid, n in nodes.items():
        if n["type"] in LM_DATASET_TYPES:
            data_nid = nid
            break
    if not data_nid:
        return {"error": "No language model dataset node found"}

    dev = get_device()

    # Encode prompt
    if not prompt:
        prompt = "\n"
    prompt_ids = [char2idx.get(ch, 0) for ch in prompt]
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

    # Set to eval mode
    for mod in trained.values():
        if hasattr(mod, 'eval'):
            mod.eval()

    tokens_info = []
    generated_chars = []

    # Build execution order (exclude optimizer/loss nodes for finding logits)
    exec_nodes = [nid for nid in order
                  if nodes[nid]["type"] not in OPTIMIZER_NODES
                  and nodes[nid]["type"] not in ALL_LOSS_NODES]

    with torch.no_grad():
        for step in range(max_tokens):
            # Feed through model
            data_inputs = {data_nid: {"out": current_ids, "labels": current_ids}}
            results = run_forward_pass(trained, nodes, edges, order, data_inputs)

            # Find logits output: [1, seq_len, vocab_size]
            logits = None
            for nid in reversed(exec_nodes):
                out = results.get(nid, {}).get("out")
                if out is not None and isinstance(out, torch.Tensor) and out.dim() == 3 and out.shape[-1] == vocab_size:
                    logits = out
                    break

            if logits is None:
                break

            # Take last position's logits
            next_logits = logits[0, -1, :] / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0 and top_k < vocab_size:
                topk_vals, topk_idx = torch.topk(next_logits, top_k)
                filtered = torch.full_like(next_logits, float('-inf'))
                filtered.scatter_(0, topk_idx, topk_vals)
                next_logits = filtered

            probs = torch.softmax(next_logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()
            prob = float(probs[next_id].item())

            char = idx2char.get(next_id, '?')
            generated_chars.append(char)
            tokens_info.append({"char": char, "prob": round(prob, 4)})

            # Append to sequence
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=dev)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

            # Limit context window
            if current_ids.shape[1] > 512:
                current_ids = current_ids[:, -512:]

    # Restore train mode
    for mod in trained.values():
        if hasattr(mod, 'train'):
            mod.train()

    generated = ''.join(generated_chars)
    return {
        "prompt": prompt,
        "generated": generated,
        "fullText": prompt + generated,
        "tokens": tokens_info,
    }
