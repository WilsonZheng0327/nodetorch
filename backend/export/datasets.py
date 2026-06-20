"""Dataset-loading code generation (image, HF text, Shakespeare, BPE)."""

import re
from export.templates import DATASET_CODE

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


def _bpe_dataset_code(data_type: str, props: dict, bpe_config: dict) -> str:
    """Generate BPE-tokenized dataset loading code."""
    vocab_size = bpe_config.get("vocabSize", 10000)
    max_len = bpe_config.get("maxLen", 256)
    batch_size = props.get("batchSize", 32)

    if data_type == "data.tiny_shakespeare":
        seq_len = props.get("seqLen", 128)
        return f'''import re, urllib.request, os
from collections import Counter

# --- BPE Tokenizer (learned from dataset) ---
class BPETokenizer:
    def __init__(self):
        self.merges, self.vocab = [], {{}}
    def train(self, text, vocab_size):
        words = re.findall(r'[a-zA-Z]+|[0-9]+|[^\\s]', text.lower())
        word_freqs = Counter()
        for w in words:
            word_freqs[tuple(w) + ('</w>',)] += 1
        self.vocab = {{'<pad>': 0, '<unk>': 1}}
        chars = set()
        for wt in word_freqs:
            chars.update(wt)
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        self.merges = []
        while len(self.vocab) < vocab_size:
            pairs = Counter()
            for wt, freq in word_freqs.items():
                for i in range(len(wt) - 1):
                    pairs[(wt[i], wt[i+1])] += freq
            if not pairs: break
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)
            new_tok = best[0] + best[1]
            self.vocab[new_tok] = len(self.vocab)
            new_freqs = {{}}
            for wt, freq in word_freqs.items():
                nw = []
                i = 0
                while i < len(wt):
                    if i < len(wt)-1 and wt[i] == best[0] and wt[i+1] == best[1]:
                        nw.append(new_tok); i += 2
                    else:
                        nw.append(wt[i]); i += 1
                new_freqs[tuple(nw)] = freq
            word_freqs = new_freqs
    def encode(self, text):
        unk = self.vocab.get('<unk>', 1)
        words = re.findall(r'[a-zA-Z]+|[0-9]+|[^\\s]', text.lower())
        ids = []
        for w in words:
            syms = list(w) + ['</w>']
            for a, b in self.merges:
                m = a + b
                i = 0
                while i < len(syms)-1:
                    if syms[i] == a and syms[i+1] == b:
                        syms[i] = m; del syms[i+1]
                    else: i += 1
            ids.extend(self.vocab.get(s, unk) for s in syms)
        return ids

# Download and tokenize
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
path = os.path.join("data", "tiny_shakespeare.txt")
os.makedirs("data", exist_ok=True)
if not os.path.exists(path):
    urllib.request.urlretrieve(url, path)
with open(path, "r") as f:
    raw_text = f.read()

print("Learning BPE merges...")
bpe = BPETokenizer()
bpe.train(raw_text, vocab_size={vocab_size})
VOCAB_SIZE = len(bpe.vocab)
print(f"BPE vocab: {{VOCAB_SIZE}} tokens")

all_ids = bpe.encode(raw_text)
data_tensor = torch.tensor(all_ids, dtype=torch.long)
SEQ_LEN = {seq_len}

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return max(1, (len(self.data) - 1) // SEQ_LEN)
    def __getitem__(self, idx):
        start = idx * SEQ_LEN
        chunk = self.data[start : start + SEQ_LEN + 1]
        if len(chunk) < SEQ_LEN + 1:
            chunk = torch.cat([chunk, torch.zeros(SEQ_LEN + 1 - len(chunk), dtype=torch.long)])
        return chunk[:-1], chunk[1:]

dataset = TextDataset(data_tensor)
train_loader = torch.utils.data.DataLoader(dataset, batch_size={batch_size}, shuffle=True)
'''
    else:
        # Classification dataset (IMDb, AG News) with BPE
        if data_type == "data.imdb":
            hf_name, num_classes = "imdb", 2
        elif data_type == "data.ag_news":
            hf_name, num_classes = "ag_news", 4
        else:
            return f"# BPE not supported for {data_type}"

        return f'''import re
from collections import Counter
from datasets import load_dataset

# --- BPE Tokenizer (learned from dataset) ---
class BPETokenizer:
    def __init__(self):
        self.merges, self.vocab = [], {{}}
    def train(self, text, vocab_size):
        words = re.findall(r'[a-zA-Z]+|[0-9]+|[^\\s]', text.lower())
        word_freqs = Counter()
        for w in words:
            word_freqs[tuple(w) + ('</w>',)] += 1
        self.vocab = {{'<pad>': 0, '<unk>': 1}}
        chars = set()
        for wt in word_freqs:
            chars.update(wt)
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        self.merges = []
        while len(self.vocab) < vocab_size:
            pairs = Counter()
            for wt, freq in word_freqs.items():
                for i in range(len(wt) - 1):
                    pairs[(wt[i], wt[i+1])] += freq
            if not pairs: break
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)
            new_tok = best[0] + best[1]
            self.vocab[new_tok] = len(self.vocab)
            new_freqs = {{}}
            for wt, freq in word_freqs.items():
                nw = []
                i = 0
                while i < len(wt):
                    if i < len(wt)-1 and wt[i] == best[0] and wt[i+1] == best[1]:
                        nw.append(new_tok); i += 2
                    else:
                        nw.append(wt[i]); i += 1
                new_freqs[tuple(nw)] = freq
            word_freqs = new_freqs
    def encode(self, text, max_len=None):
        unk = self.vocab.get('<unk>', 1)
        words = re.findall(r'[a-zA-Z]+|[0-9]+|[^\\s]', text.lower())
        ids = []
        for w in words:
            syms = list(w) + ['</w>']
            for a, b in self.merges:
                m = a + b
                i = 0
                while i < len(syms)-1:
                    if syms[i] == a and syms[i+1] == b:
                        syms[i] = m; del syms[i+1]
                    else: i += 1
            ids.extend(self.vocab.get(s, unk) for s in syms)
        if max_len:
            ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
        return ids

NUM_CLASSES = {num_classes}
VOCAB_SIZE = {vocab_size}
MAX_LEN = {max_len}

ds = load_dataset("{hf_name}", split="train")
print("Learning BPE merges...")
bpe = BPETokenizer()
corpus = "\\n".join(ds["text"][:5000])
bpe.train(corpus, vocab_size=VOCAB_SIZE)
VOCAB_SIZE = len(bpe.vocab)
print(f"BPE vocab: {{VOCAB_SIZE}} tokens")

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        ids = bpe.encode(self.ds[idx]["text"], max_len=MAX_LEN)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.ds[idx]["label"], dtype=torch.long)

dataset = TextDataset(ds)
train_loader = torch.utils.data.DataLoader(dataset, batch_size={batch_size}, shuffle=True)
'''


def _generate_dataset_code(data_node: dict, bpe_config: dict | None = None) -> str:
    """Generate dataset loading code."""
    dtype = data_node["type"]
    props = data_node.get("properties", {})
    batch_size = props.get("batchSize", 32)

    # BPE mode — generate BPE tokenizer + dataset code
    if bpe_config:
        return _bpe_dataset_code(dtype, props, bpe_config)

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


