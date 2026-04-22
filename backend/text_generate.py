"""Text generation from a trained autoregressive model.

Iterative decode loop: encode prompt → forward → sample next token → append → repeat.
Supports temperature scaling and top-k sampling.
Detects tokenizer mode (character or BPE) from the graph and uses the right encoding.
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
from data_loaders import get_shakespeare_vocab, LM_DATASET_TYPES, get_raw_texts
from bpe import get_bpe_tokenizer

# Node types that should be bypassed during generation (not part of the model forward pass)
_GENERATION_SKIP_TYPES = ALL_LOSS_NODES | set(OPTIMIZER_NODES) | {"ml.preprocessing.tokenizer"}


def _get_tokenizer_config(nodes: dict) -> dict | None:
    """Find the tokenizer node and return its properties, or None."""
    for n in nodes.values():
        if n.get("type") == "ml.preprocessing.tokenizer":
            return n.get("properties", {})
    return None


def _get_dataset_type(nodes: dict) -> str | None:
    """Find the LM dataset node type."""
    for n in nodes.values():
        if n.get("type") in LM_DATASET_TYPES:
            return n["type"]
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
        { prompt, generated, fullText, tokens: [{token, prob}] }
    """
    if not has_trained_model():
        return {"error": "No trained model — train first"}

    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    # Find data node
    data_nid = None
    for nid, n in nodes.items():
        if n["type"] in LM_DATASET_TYPES:
            data_nid = nid
            break
    if not data_nid:
        return {"error": "No language model dataset node found"}

    # Detect tokenization mode from graph
    tok_config = _get_tokenizer_config(nodes)
    tok_mode = tok_config.get("mode", "character") if tok_config else "character"
    dataset_type = _get_dataset_type(nodes)

    # Build encode/decode functions based on tokenizer mode
    if tok_mode == "bpe" and dataset_type:
        bpe_vocab_size = tok_config.get("vocabSize", 500) if tok_config else 500
        raw_text = get_raw_texts(dataset_type)
        bpe = get_bpe_tokenizer(raw_text, bpe_vocab_size, cache_key=dataset_type)
        vocab_size = bpe.vocab_size

        def encode_prompt(text: str) -> list[int]:
            return bpe.encode(text)

        def decode_token(token_id: int) -> str:
            return bpe.decode_token(token_id)
    else:
        # Character-level (default)
        char2idx, idx2char = get_shakespeare_vocab()
        vocab_size = len(char2idx)

        def encode_prompt(text: str) -> list[int]:
            return [char2idx.get(ch, 0) for ch in text]

        def decode_token(token_id: int) -> str:
            return idx2char.get(token_id, '?')

    # Context window = positional encoding's max_len
    max_context = 512
    for n in nodes.values():
        if n["type"] == "ml.layers.positional_encoding":
            max_context = n.get("properties", {}).get("maxLen", 512)
            break

    dev = get_device()

    # Encode prompt
    if not prompt:
        prompt = "\n" if tok_mode != "bpe" else "the"
    prompt_ids = encode_prompt(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

    # Set to eval mode
    for mod in trained.values():
        if hasattr(mod, 'eval'):
            mod.eval()

    tokens_info = []
    generated_tokens = []

    # Build execution order (exclude optimizer/loss nodes for finding logits)
    exec_nodes = [nid for nid in order
                  if nodes[nid]["type"] not in OPTIMIZER_NODES
                  and nodes[nid]["type"] not in ALL_LOSS_NODES]

    # Filter out loss/optimizer/tokenizer modules — they must not run during generation.
    gen_modules = {k: v for k, v in trained.items()
                   if nodes.get(k, {}).get("type") not in _GENERATION_SKIP_TYPES}

    with torch.no_grad():
        for step in range(max_tokens):
            data_inputs = {data_nid: {"out": current_ids, "labels": current_ids}}
            # Pre-populate tokenizer results so downstream nodes get unpadded IDs
            for nid, n in nodes.items():
                if n["type"] == "ml.preprocessing.tokenizer":
                    data_inputs[nid] = {"out": current_ids}
            results = run_forward_pass(gen_modules, nodes, edges, order, data_inputs)

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

            token_str = decode_token(next_id)
            generated_tokens.append(token_str)
            tokens_info.append({"char": token_str, "prob": round(prob, 4)})

            # Append to sequence
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=dev)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

            # Limit context window to the positional encoding's max_len
            if current_ids.shape[1] > max_context:
                current_ids = current_ids[:, -max_context:]

    # Restore train mode
    for mod in trained.values():
        if hasattr(mod, 'train'):
            mod.train()

    generated = ''.join(generated_tokens)
    return {
        "prompt": prompt,
        "generated": generated,
        "fullText": prompt + generated,
        "tokens": tokens_info,
    }
