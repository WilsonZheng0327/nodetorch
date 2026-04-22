# data_loaders.py — Per-dataset loader functions.
#
# ALL dataset-specific knowledge lives here — normalization constants,
# dataset construction, denormalization for image preview. graph_builder.py
# calls these generically via four registries:
#
#   DATA_LOADERS:     (props) → { port_id: tensor }     — load a batch
#   TRAIN_DATASETS:   () → Dataset                       — full training dataset
#   DENORMALIZERS:    (tensor [C,H,W]) → tensor [C,H,W]  — undo normalization for preview
#   DATASET_DETAILS:  () → dict                           — labels, sample images, stats
#
# Adding a new dataset: implement 4 functions, add to 4 registries.
# No changes needed in graph_builder.py.

import torch
import torchvision
import torchvision.transforms as transforms

# All downloaded datasets go here
DATASETS_DIR = "./storage/datasets"

# Dataset types that signal autoregressive/language model training mode
LM_DATASET_TYPES = {"data.tiny_shakespeare"}


# --- MNIST ---

def load_mnist(props: dict) -> dict[str, torch.Tensor]:
    """Load a batch from MNIST."""
    batch_size = props.get("batchSize", 1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = torchvision.datasets.MNIST(
        root=DATASETS_DIR, train=True, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))
    return {"out": images, "labels": labels}


def train_dataset_mnist() -> torch.utils.data.Dataset:
    """Return the full MNIST training dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return torchvision.datasets.MNIST(
        root=DATASETS_DIR, train=True, download=True, transform=transform,
    )


def denormalize_mnist(img: torch.Tensor) -> torch.Tensor:
    """Undo MNIST normalization. Input: [C, H, W], output: [C, H, W] in 0-1 range."""
    return img * 0.3081 + 0.1307


# --- CIFAR-100 ---

def load_cifar100(props: dict) -> dict[str, torch.Tensor]:
    """Load a batch from CIFAR-100."""
    batch_size = props.get("batchSize", 32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])
    dataset = torchvision.datasets.CIFAR100(
        root=DATASETS_DIR, train=True, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))
    return {"out": images, "labels": labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)}


def train_dataset_cifar100(augHFlip: bool = False, augRandomCrop: bool = False, augColorJitter: bool = False) -> torch.utils.data.Dataset:
    """Return the full CIFAR-100 training dataset, with optional augmentations."""
    aug_list: list = []
    if augRandomCrop:
        aug_list.append(transforms.RandomCrop(32, padding=4))
    if augHFlip:
        aug_list.append(transforms.RandomHorizontalFlip())
    if augColorJitter:
        aug_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    transform = transforms.Compose([
        *aug_list,
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])
    return torchvision.datasets.CIFAR100(
        root=DATASETS_DIR, train=True, download=True, transform=transform,
    )


def denormalize_cifar100(img: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-100 normalization. Input: [C, H, W], output: [C, H, W] in 0-1 range."""
    mean = torch.tensor([0.5071, 0.4867, 0.4408], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.2675, 0.2565, 0.2761], device=img.device).view(3, 1, 1)
    return img * std + mean


# --- CIFAR-10 ---

def load_cifar10(props: dict) -> dict[str, torch.Tensor]:
    batch_size = props.get("batchSize", 32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=DATASETS_DIR, train=True, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))
    return {"out": images, "labels": labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)}


def train_dataset_cifar10(augHFlip: bool = False, augRandomCrop: bool = False, augColorJitter: bool = False) -> torch.utils.data.Dataset:
    aug_list: list = []
    if augRandomCrop:
        aug_list.append(transforms.RandomCrop(32, padding=4))
    if augHFlip:
        aug_list.append(transforms.RandomHorizontalFlip())
    if augColorJitter:
        aug_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    transform = transforms.Compose([
        *aug_list,
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return torchvision.datasets.CIFAR10(
        root=DATASETS_DIR, train=True, download=True, transform=transform,
    )


def denormalize_cifar10(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=img.device).view(3, 1, 1)
    return img * std + mean


# --- FashionMNIST ---

def load_fashion_mnist(props: dict) -> dict[str, torch.Tensor]:
    batch_size = props.get("batchSize", 32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    dataset = torchvision.datasets.FashionMNIST(
        root=DATASETS_DIR, train=True, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))
    return {"out": images, "labels": labels}


def train_dataset_fashion_mnist() -> torch.utils.data.Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    return torchvision.datasets.FashionMNIST(
        root=DATASETS_DIR, train=True, download=True, transform=transform,
    )


def denormalize_fashion_mnist(img: torch.Tensor) -> torch.Tensor:
    return img * 0.3530 + 0.2860


# --- Text dataset utilities ---

import re
from collections import Counter

# Simple word-level tokenizer (lowercase, split on non-alpha)
def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z]+', text.lower())


# Cache built vocabs so they're only computed once
_vocab_cache: dict[str, dict[str, int]] = {}


def _build_vocab(texts: list[str], vocab_size: int, cache_key: str) -> dict[str, int]:
    """Build a word→index mapping from texts. Cached per cache_key."""
    full_key = f"{cache_key}_{vocab_size}"
    if full_key in _vocab_cache:
        return _vocab_cache[full_key]

    counter: Counter = Counter()
    for text in texts:
        counter.update(_tokenize(text))

    # Reserve 0=pad, 1=unknown
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)

    _vocab_cache[full_key] = vocab
    return vocab


def _encode_texts(texts: list[str], vocab: dict[str, int], max_len: int) -> torch.Tensor:
    """Tokenize and encode a list of texts to padded integer tensor."""
    unk = vocab.get("<unk>", 1)
    encoded = []
    for text in texts:
        tokens = _tokenize(text)
        ids = [vocab.get(t, unk) for t in tokens[:max_len]]
        # Pad to max_len
        ids = ids + [0] * (max_len - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


# --- IMDb ---

_imdb_cache: dict[str, object] = {}

def _get_imdb_dataset():
    if "ds" not in _imdb_cache:
        from datasets import load_dataset
        _imdb_cache["ds"] = load_dataset("imdb", split="train")
    return _imdb_cache["ds"]


def load_imdb(props: dict) -> dict[str, torch.Tensor]:
    batch_size = props.get("batchSize", 32)
    max_len = props.get("maxLen", 256)
    vocab_size = props.get("vocabSize", 10000)

    ds = _get_imdb_dataset()
    vocab = _build_vocab(ds["text"][:5000], vocab_size, "imdb")  # Build from first 5K for speed

    # Sample a batch
    indices = torch.randperm(len(ds))[:batch_size].tolist()
    texts = [ds[i]["text"] for i in indices]
    labels = torch.tensor([ds[i]["label"] for i in indices], dtype=torch.long)
    tokens = _encode_texts(texts, vocab, max_len)

    return {"out": tokens, "labels": labels, "_texts": texts}


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=10000, max_len=256, split="train"):
        if split == "test":
            from datasets import load_dataset
            self.ds = load_dataset("imdb", split="test")
        else:
            self.ds = _get_imdb_dataset()
        # Always build vocab from train set for consistency
        train_ds = _get_imdb_dataset()
        self.vocab = _build_vocab(train_ds["text"][:5000], vocab_size, "imdb")
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["text"]
        label = self.ds[idx]["label"]
        unk = self.vocab.get("<unk>", 1)
        tokens = _tokenize(text)
        ids = [self.vocab.get(t, unk) for t in tokens[:self.max_len]]
        ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def train_dataset_imdb(vocab_size=10000, max_len=256) -> torch.utils.data.Dataset:
    return IMDbDataset(vocab_size, max_len)


# --- AG News ---

_agnews_cache: dict[str, object] = {}

def _get_agnews_dataset():
    if "ds" not in _agnews_cache:
        from datasets import load_dataset
        _agnews_cache["ds"] = load_dataset("ag_news", split="train")
    return _agnews_cache["ds"]


def load_ag_news(props: dict) -> dict[str, torch.Tensor]:
    batch_size = props.get("batchSize", 32)
    max_len = props.get("maxLen", 128)
    vocab_size = props.get("vocabSize", 10000)

    ds = _get_agnews_dataset()
    vocab = _build_vocab(ds["text"][:5000], vocab_size, "agnews")

    indices = torch.randperm(len(ds))[:batch_size].tolist()
    texts = [ds[i]["text"] for i in indices]
    labels = torch.tensor([ds[i]["label"] for i in indices], dtype=torch.long)
    tokens = _encode_texts(texts, vocab, max_len)

    return {"out": tokens, "labels": labels, "_texts": texts}


class AGNewsDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size=10000, max_len=128, split="train"):
        if split == "test":
            from datasets import load_dataset
            self.ds = load_dataset("ag_news", split="test")
        else:
            self.ds = _get_agnews_dataset()
        train_ds = _get_agnews_dataset()
        self.vocab = _build_vocab(train_ds["text"][:5000], vocab_size, "agnews")
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx]["text"]
        label = self.ds[idx]["label"]
        unk = self.vocab.get("<unk>", 1)
        tokens = _tokenize(text)
        ids = [self.vocab.get(t, unk) for t in tokens[:self.max_len]]
        ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def train_dataset_ag_news(vocab_size=10000, max_len=128) -> torch.utils.data.Dataset:
    return AGNewsDataset(vocab_size, max_len)


# --- TinyShakespeare (character-level language modeling) ---

import os
import urllib.request

_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
_shakespeare_cache: dict[str, object] = {}


def _get_shakespeare_text() -> str:
    """Load or download TinyShakespeare text."""
    if "text" in _shakespeare_cache:
        return _shakespeare_cache["text"]
    filepath = os.path.join(DATASETS_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(filepath):
        os.makedirs(DATASETS_DIR, exist_ok=True)
        urllib.request.urlretrieve(_SHAKESPEARE_URL, filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    _shakespeare_cache["text"] = text
    return text


def get_shakespeare_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Get character-level vocab (char→idx, idx→char). Cached."""
    if "char2idx" in _shakespeare_cache:
        return _shakespeare_cache["char2idx"], _shakespeare_cache["idx2char"]
    text = _get_shakespeare_text()
    chars = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    _shakespeare_cache["char2idx"] = char2idx
    _shakespeare_cache["idx2char"] = idx2char
    return char2idx, idx2char


class TinyShakespeareDataset(torch.utils.data.Dataset):
    """Character-level LM dataset. Each sample is (input_ids, target_ids) where
    target is shifted by 1 position (next-character prediction)."""

    def __init__(self, seq_len: int = 128):
        text = _get_shakespeare_text()
        self.char2idx, _ = get_shakespeare_vocab()
        self.data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        # If chunk is short (end of text), pad
        if len(chunk) < self.seq_len + 1:
            chunk = torch.cat([chunk, torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)])
        return chunk[:-1], chunk[1:]  # (input, target)


def load_tiny_shakespeare(props: dict) -> dict[str, torch.Tensor]:
    """Load a batch from TinyShakespeare."""
    batch_size = props.get("batchSize", 32)
    seq_len = props.get("seqLen", 128)
    dataset = TinyShakespeareDataset(seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    inputs, targets = next(iter(loader))
    return {"out": inputs, "labels": targets}


def train_dataset_tiny_shakespeare(seqLen: int = 128) -> torch.utils.data.Dataset:
    return TinyShakespeareDataset(seqLen)


def detail_tiny_shakespeare() -> dict:
    text = _get_shakespeare_text()
    char2idx, _ = get_shakespeare_vocab()
    sample = text[:500]
    return {
        "name": "Tiny Shakespeare",
        "description": "Character-level language modeling on Shakespeare plays (~1MB)",
        "isText": True,
        "isLanguageModel": True,
        "labels": [],
        "trainSamples": len(text),
        "vocabSize": len(char2idx),
        "diskSize": "~1 MB",
        "sampleTexts": {"Sample": [sample]},
    }


# --- BPE dataset wrappers ---
# When the tokenizer node is in BPE mode, these replace the default datasets.
# BPE merge rules are learned from the training text before the training loop.

from bpe import get_bpe_tokenizer, BPETokenizer


def get_raw_texts(dataset_type: str) -> str:
    """Get raw text corpus for a dataset (used for BPE training)."""
    if dataset_type == "data.tiny_shakespeare":
        return _get_shakespeare_text()
    if dataset_type == "data.imdb":
        ds = _get_imdb_dataset()
        return '\n'.join(ds["text"][:5000])  # first 5K for speed
    if dataset_type == "data.ag_news":
        ds = _get_agnews_dataset()
        return '\n'.join(ds["text"][:5000])
    return ""


class BPETextDataset(torch.utils.data.Dataset):
    """Classification dataset with BPE tokenization (IMDb, AG News)."""

    def __init__(self, dataset_type: str, bpe: BPETokenizer, max_len: int = 256):
        if dataset_type == "data.imdb":
            self.ds = _get_imdb_dataset()
            self.text_key = "text"
            self.label_key = "label"
        elif dataset_type == "data.ag_news":
            self.ds = _get_agnews_dataset()
            self.text_key = "text"
            self.label_key = "label"
        else:
            raise ValueError(f"BPE not supported for {dataset_type}")
        self.bpe = bpe
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = self.ds[idx][self.text_key]
        label = self.ds[idx][self.label_key]
        ids = self.bpe.encode(text, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class BPELMDataset(torch.utils.data.Dataset):
    """Language modeling dataset with BPE tokenization (next-token prediction)."""

    def __init__(self, raw_text: str, bpe: BPETokenizer, seq_len: int = 128):
        self.seq_len = seq_len
        # Encode entire corpus once
        all_ids = bpe.encode(raw_text)
        self.data = torch.tensor(all_ids, dtype=torch.long)

    def __len__(self):
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = torch.cat([chunk, torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)])
        return chunk[:-1], chunk[1:]


# --- Registries ---
# Each registry maps node type string → function.
# graph_builder.py uses these — no if/else chains needed.

# Load a batch: (props) → { port_id: tensor }
DATA_LOADERS: dict[str, callable] = {
    "data.mnist": load_mnist,
    "data.cifar10": load_cifar10,
    "data.cifar100": load_cifar100,
    "data.fashion_mnist": load_fashion_mnist,
    "data.imdb": load_imdb,
    "data.ag_news": load_ag_news,
    "data.tiny_shakespeare": load_tiny_shakespeare,
}

# Get full training dataset: () → Dataset
TRAIN_DATASETS: dict[str, callable] = {
    "data.mnist": train_dataset_mnist,
    "data.cifar10": train_dataset_cifar10,
    "data.cifar100": train_dataset_cifar100,
    "data.fashion_mnist": train_dataset_fashion_mnist,
    "data.imdb": train_dataset_imdb,
    "data.ag_news": train_dataset_ag_news,
    "data.tiny_shakespeare": train_dataset_tiny_shakespeare,
}

# Undo normalization for image preview: (tensor [C,H,W]) → tensor [C,H,W] in 0-1
DENORMALIZERS: dict[str, callable] = {
    "data.mnist": denormalize_mnist,
    "data.cifar10": denormalize_cifar10,
    "data.cifar100": denormalize_cifar100,
    "data.fashion_mnist": denormalize_fashion_mnist,
}


# --- Dataset detail info ---
# Returns { labels: [...], samplesPerClass: int, imageSize: [H,W], channels: int, ... }

def _tensor_to_pixels(img: torch.Tensor, denorm) -> list:
    """Convert a [C,H,W] tensor to pixel data for JSON."""
    if denorm:
        img = denorm(img)
    img = (img.clamp(0, 1) * 255).byte()
    if img.shape[0] == 1:
        return img[0].tolist()
    return img.permute(1, 2, 0).tolist()


def detail_mnist() -> dict:
    labels = [str(i) for i in range(10)]
    dataset = torchvision.datasets.MNIST(
        root=DATASETS_DIR, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    # Get 4 samples per class
    samples: dict[int, list] = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(samples[label]) < 4:
            # Normalize then denormalize for consistency
            normed = transforms.Normalize((0.1307,), (0.3081,))(img)
            samples[label].append(_tensor_to_pixels(normed, denormalize_mnist))
        if all(len(v) >= 4 for v in samples.values()):
            break

    return {
        "name": "MNIST",
        "description": "Handwritten digits, 28x28 grayscale",
        "labels": labels,
        "channels": 1,
        "imageSize": [28, 28],
        "trainSamples": 60000,
        "testSamples": 10000,
        "diskSize": "~12 MB",
        "sampleImages": {str(k): v for k, v in samples.items()},
    }


def detail_cifar100() -> dict:
    fine_labels = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm',
    ]
    coarse_labels = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects',
        'large_carnivores', 'large_man-made_outdoor_things',
        'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles',
        'small_mammals', 'trees', 'vehicles_1', 'vehicles_2',
    ]
    dataset = torchvision.datasets.CIFAR100(
        root=DATASETS_DIR, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    # Get 4 samples for first 20 classes (all 100 would be too heavy)
    max_classes = 20
    samples: dict[int, list] = {i: [] for i in range(max_classes)}
    for img, label in dataset:
        if label < max_classes and len(samples[label]) < 4:
            normed = transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            )(img)
            samples[label].append(_tensor_to_pixels(normed, denormalize_cifar100))
        if all(len(v) >= 4 for v in samples.values()):
            break

    return {
        "name": "CIFAR-100",
        "description": "100-class color images, 32x32 RGB",
        "labels": fine_labels,
        "coarseLabels": coarse_labels,
        "channels": 3,
        "imageSize": [32, 32],
        "trainSamples": 50000,
        "testSamples": 10000,
        "diskSize": "~161 MB",
        "sampleImages": {fine_labels[k]: v for k, v in samples.items()},
    }


def detail_cifar10() -> dict:
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dataset = torchvision.datasets.CIFAR10(
        root=DATASETS_DIR, train=True, download=True, transform=transforms.ToTensor(),
    )
    samples: dict[int, list] = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(samples[label]) < 4:
            normed = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))(img)
            samples[label].append(_tensor_to_pixels(normed, denormalize_cifar10))
        if all(len(v) >= 4 for v in samples.values()):
            break
    return {
        "name": "CIFAR-10",
        "description": "10-class color images, 32x32 RGB",
        "labels": labels,
        "channels": 3,
        "imageSize": [32, 32],
        "trainSamples": 50000,
        "testSamples": 10000,
        "diskSize": "~163 MB",
        "sampleImages": {labels[k]: v for k, v in samples.items()},
    }


def detail_fashion_mnist() -> dict:
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    dataset = torchvision.datasets.FashionMNIST(
        root=DATASETS_DIR, train=True, download=True, transform=transforms.ToTensor(),
    )
    samples: dict[int, list] = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(samples[label]) < 4:
            normed = transforms.Normalize((0.2860,), (0.3530,))(img)
            samples[label].append(_tensor_to_pixels(normed, denormalize_fashion_mnist))
        if all(len(v) >= 4 for v in samples.values()):
            break
    return {
        "name": "FashionMNIST",
        "description": "Fashion items, 28x28 grayscale, 10 classes",
        "labels": labels,
        "channels": 1,
        "imageSize": [28, 28],
        "trainSamples": 60000,
        "testSamples": 10000,
        "diskSize": "~30 MB",
        "sampleImages": {labels[k]: v for k, v in samples.items()},
    }


def detail_imdb() -> dict:
    ds = _get_imdb_dataset()
    # Sample texts per class
    samples: dict[int, list] = {0: [], 1: []}
    for item in ds:
        label = item["label"]
        if len(samples[label]) < 3:
            text = item["text"][:200] + ("..." if len(item["text"]) > 200 else "")
            samples[label].append(text)
        if all(len(v) >= 3 for v in samples.values()):
            break
    return {
        "name": "IMDb",
        "description": "Movie review sentiment classification (binary)",
        "labels": ["Negative", "Positive"],
        "trainSamples": len(ds),
        "diskSize": "~80 MB",
        "isText": True,
        "sampleTexts": {["Negative", "Positive"][k]: v for k, v in samples.items()},
    }


def detail_ag_news() -> dict:
    ds = _get_agnews_dataset()
    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    samples: dict[int, list] = {i: [] for i in range(4)}
    for item in ds:
        label = item["label"]
        if len(samples[label]) < 3:
            text = item["text"][:200] + ("..." if len(item["text"]) > 200 else "")
            samples[label].append(text)
        if all(len(v) >= 3 for v in samples.values()):
            break
    return {
        "name": "AG News",
        "description": "News article classification (4 classes)",
        "labels": label_names,
        "trainSamples": len(ds),
        "diskSize": "~30 MB",
        "isText": True,
        "sampleTexts": {label_names[k]: v for k, v in samples.items()},
    }


# Get dataset detail info: () → dict
DATASET_DETAILS: dict[str, callable] = {
    "data.mnist": detail_mnist,
    "data.cifar10": detail_cifar10,
    "data.cifar100": detail_cifar100,
    "data.fashion_mnist": detail_fashion_mnist,
    "data.imdb": detail_imdb,
    "data.ag_news": detail_ag_news,
    "data.tiny_shakespeare": detail_tiny_shakespeare,
}

# Class name lookup per dataset — used for confusion matrix labels, etc.
CLASS_NAMES: dict[str, list[str]] = {
    "data.mnist": [str(i) for i in range(10)],
    "data.fashion_mnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
    "data.cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "data.cifar100": [str(i) for i in range(100)],  # too many for labels, use indices
    "data.imdb": ["Negative", "Positive"],
    "data.ag_news": ["World", "Sports", "Business", "Sci/Tech"],
}


# --- Test datasets (held-out test split, never seen during training) ---

def test_dataset_mnist() -> torch.utils.data.Dataset:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return torchvision.datasets.MNIST(root=DATASETS_DIR, train=False, download=True, transform=transform)

def test_dataset_cifar10(**kwargs) -> torch.utils.data.Dataset:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    return torchvision.datasets.CIFAR10(root=DATASETS_DIR, train=False, download=True, transform=transform)

def test_dataset_cifar100(**kwargs) -> torch.utils.data.Dataset:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    return torchvision.datasets.CIFAR100(root=DATASETS_DIR, train=False, download=True, transform=transform)

def test_dataset_fashion_mnist() -> torch.utils.data.Dataset:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    return torchvision.datasets.FashionMNIST(root=DATASETS_DIR, train=False, download=True, transform=transform)

def test_dataset_imdb(**kwargs) -> torch.utils.data.Dataset:
    return IMDbDataset(vocab_size=kwargs.get("vocabSize", 10000), max_len=kwargs.get("maxLen", 256), split="test")

def test_dataset_ag_news(**kwargs) -> torch.utils.data.Dataset:
    return AGNewsDataset(vocab_size=kwargs.get("vocabSize", 10000), max_len=kwargs.get("maxLen", 128), split="test")

TEST_DATASETS: dict[str, callable] = {
    "data.mnist": test_dataset_mnist,
    "data.cifar10": test_dataset_cifar10,
    "data.cifar100": test_dataset_cifar100,
    "data.fashion_mnist": test_dataset_fashion_mnist,
    "data.imdb": test_dataset_imdb,
    "data.ag_news": test_dataset_ag_news,
}


# --- Augmentation preview ---

def _get_raw_image(dataset_type: str):
    """Get the first raw sample (PIL image) from a dataset, no transforms."""
    if dataset_type == "data.cifar10":
        ds = torchvision.datasets.CIFAR10(root=DATASETS_DIR, train=True, download=True)
        return ds[0][0]  # PIL image
    if dataset_type == "data.cifar100":
        ds = torchvision.datasets.CIFAR100(root=DATASETS_DIR, train=True, download=True)
        return ds[0][0]
    if dataset_type == "data.mnist":
        ds = torchvision.datasets.MNIST(root=DATASETS_DIR, train=True, download=True)
        return ds[0][0]
    if dataset_type == "data.fashion_mnist":
        ds = torchvision.datasets.FashionMNIST(root=DATASETS_DIR, train=True, download=True)
        return ds[0][0]
    return None


def _pil_to_pixels(img) -> tuple[list, int]:
    """Convert a PIL image (or tensor) to a (pixels, channels) tuple for display."""
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)  # [C, H, W] in [0, 1]
    img = (img.clamp(0, 1) * 255).byte()
    C = img.shape[0]
    if C == 1:
        return img[0].tolist(), 1
    return img.permute(1, 2, 0).tolist(), int(C)


def augmentation_preview(
    dataset_type: str,
    augHFlip: bool = False,
    augRandomCrop: bool = False,
    augColorJitter: bool = False,
    num_variants: int = 8,
) -> dict | None:
    """Generate augmented variants of a sample with the given augmentation settings.

    Returns the original sample + num_variants augmented versions (different random seeds).
    """
    pil_img = _get_raw_image(dataset_type)
    if pil_img is None:
        return {"error": f"Augmentation preview not supported for {dataset_type}"}

    # Build augmentation pipeline (same as training, but in PIL space + ToTensor at end)
    pil_aug_list = []
    if augRandomCrop:
        pil_aug_list.append(transforms.RandomCrop(pil_img.size[0], padding=4))
    if augHFlip:
        pil_aug_list.append(transforms.RandomHorizontalFlip())
    if augColorJitter:
        pil_aug_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    original_pixels, channels = _pil_to_pixels(pil_img)
    original = {"pixels": original_pixels, "channels": channels}

    if not pil_aug_list:
        return {"original": original, "variants": [], "anyEnabled": False}

    variants = []
    for i in range(num_variants):
        # Seed per variant so they differ visibly
        torch.manual_seed(i + 1)
        import random as _random
        _random.seed(i + 1)
        augmented = pil_img
        for t in pil_aug_list:
            augmented = t(augmented)
        pixels, c = _pil_to_pixels(augmented)
        variants.append({"pixels": pixels, "channels": c})

    return {"original": original, "variants": variants, "anyEnabled": True}
