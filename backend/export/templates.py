"""Reusable code-string constants injected into generated scripts."""


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


_TOKENIZER_CLASS = '''class Tokenizer(nn.Module):
    """Truncates/pads sequences to max_len and caps vocabulary indices.

    Ensures all token ID sequences have uniform length and valid indices
    before being fed to an embedding layer.
    """
    def __init__(self, vocab_size, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

    def forward(self, x):
        x = x[:, :self.max_len]
        if x.shape[1] < self.max_len:
            x = torch.nn.functional.pad(x, (0, self.max_len - x.shape[1]))
        return x.clamp(0, self.vocab_size - 1)
'''


_POSITIONAL_ENCODING_CLASS = '''class PositionalEncoding(nn.Module):
    """Adds position information to token embeddings.

    'learned': trainable per-position embeddings (GPT/BERT style).
    'sinusoidal': fixed sin/cos waves (original Transformer paper).
    Input/output shape: [B, seq_len, embed_dim].
    """
    def __init__(self, max_len, embed_dim, encoding_type='learned'):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.encoding_type = encoding_type
        if encoding_type == 'learned':
            self.pos_embed = nn.Embedding(max_len, embed_dim)
        else:
            import math
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            cos_part = torch.cos(position * div_term)
            pe[:, 1::2] = cos_part[:, : pe[:, 1::2].shape[1]]
            self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.shape[1]
        if self.encoding_type == 'learned':
            positions = torch.arange(seq_len, device=x.device)
            pos = self.pos_embed(positions)
        else:
            pos = self.pe[:seq_len].to(x.device)
        return x + pos.unsqueeze(0)
'''


