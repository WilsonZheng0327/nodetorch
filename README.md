<p align="center">
  <img src="public/nodetorch.png" alt="NodeTorch" width="120" />
</p>
<h1 align="center">NodeTorch</h1>
<p align="center">Node-based visual educational tool for building, inspecting, and understanding ML models.

---

## Demo

https://github.com/user-attachments/assets/a58c5c44-d274-4949-8a00-f989c418fb07

---

Build neural networks by dragging nodes onto a canvas and connecting them. Shape inference runs instantly as you edit. Train with real PyTorch, visualize every layer, and step through forward and backward passes to understand what your model is doing.

## Features

- **Visual graph editor** — drag-and-drop nodes, auto-wiring, multi-select, copy/paste, undo/redo
- **Instant shape inference** — output shapes update live as you connect and edit
- **Real PyTorch training** — train on GPU/CPU with live loss/accuracy charts, per-class accuracy, gradient flow, confusion matrix
- **Forward step-through** — walk through each layer's transformation one step at a time, see feature maps, activations, stats, and plain-English insights
- **Backward step-through** — visualize gradients flowing backward, see gradient kernels, spatial heatmaps, per-neuron bars, vanishing/exploding gradient detection
- **Inference** — run trained models on new samples, see predictions (classification) or reconstructions (autoencoders)
- **Loss landscape** — 2D visualization of the loss surface around trained weights
- **Node visualization** — per-node weight/gradient/activation histograms with health indicators
- **Text generation** — train language models (character-level or BPE subword), then generate text interactively with temperature and top-K sampling
- **Save/load** — export graphs as JSON, save/load trained weights to disk
- **Export to Python** — generate a standalone, runnable PyTorch training script from any graph
- **Model presets** — one-click load of 17 pre-built architectures
- **Guided tutorials** — built-in tutorial panel with 8 tracks (UI basics + paradigm-specific walkthroughs for CNNs, VAEs, GANs, diffusion, and language models)
- **Dark/light theme** — toggle in bottom-left corner

## Supported Architectures

| Task | Datasets | Example Presets |
|------|----------|----------------|
| Image Classification | MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100 | MLP, LeNet-5, VGG, ResNet-34 |
| Text Classification | IMDb, AG News | LSTM, Self-Attention |
| Transfer Learning | CIFAR-10 | Pretrained ResNet-18 |
| Reconstruction | MNIST | Convolutional Autoencoder, VAE |
| Generative (GAN) | MNIST | DCGAN |
| Generative (Diffusion) | MNIST | Diffusion, Diffusion U-Net |
| Language Modeling | Tiny Shakespeare (chars) | Char-LM (LSTM), Mini-GPT (Transformer) |

## Available Nodes

**Layers**: Conv2d, Conv1d, ConvTranspose2d, Linear, Embedding, PositionalEncoding, LSTM, GRU, RNN, MultiHeadAttention, Attention, Flatten, Upsample, Pretrained ResNet-18

**Preprocessing**: Tokenizer (character / word / BPE)

**Normalization**: BatchNorm2d/1d, LayerNorm, GroupNorm, InstanceNorm2d

**Activations**: ReLU, Sigmoid, Tanh, Softmax, GELU, LeakyReLU

**Pooling**: MaxPool2d/1d, AvgPool2d, AdaptiveAvgPool2d

**Structural**: Add (residual), Concat, Reshape, Permute, SequencePool, Reparameterize, Subgraph Blocks

**Loss**: CrossEntropy, MSE, VAE Loss, GAN Loss

**GAN**: Noise Input

**Diffusion**: Noise Scheduler

**Optimizers**: SGD, Adam, AdamW — with schedulers (cosine, step, warmup), early stopping, gradient clipping

## Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+
- **PyTorch** — GPU strongly recommended for training. CPU works but is much slower.

> **CUDA version matters.** If you have an NVIDIA GPU, your PyTorch CUDA version must match your installed NVIDIA driver. Check your driver version with `nvidia-smi`, then pick the matching PyTorch install command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). Mismatched versions will silently fall back to CPU.

## Setup

### Option 1: Start script (recommended)

The start script auto-detects your GPU, installs everything, and launches both servers.

**macOS / Linux:**
```bash
chmod +x start.sh
./start.sh
```

**Windows:**
```
start.bat
```

First run takes 2-5 minutes (downloads PyTorch + dependencies). After that, starts in seconds. Press Ctrl+C to stop both servers.

### Option 2: Manual setup (two terminals)

Use this if the start script doesn't work, or if you want more control.

**Terminal 1 — Backend:**
```bash
# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install PyTorch — pick ONE based on your hardware:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128    # NVIDIA GPU (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124    # NVIDIA GPU (CUDA 12.4)
pip install torch torchvision                                                        # Apple Silicon (MPS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu      # CPU only

# Install remaining dependencies
pip install -r requirements.txt

# Start backend server
cd backend
python main.py
```

Backend runs at http://localhost:8000.

**Terminal 2 — Frontend:**
```bash
npm install
npm run dev
```

Frontend opens at http://localhost:5173.

> **Verify GPU is detected:** When the backend starts, it logs the device (e.g. `CUDA: NVIDIA RTX 4090`). If it says `CPU only` but you have a GPU, your PyTorch CUDA version likely doesn't match your driver — reinstall PyTorch with the correct CUDA version.

## Usage

1. **Add nodes** — drag from the palette (Tab to toggle) onto the canvas
2. **Connect nodes** — drag from an output port to an input port
3. **Edit properties** — click a node, edit in the inspector (right panel)
4. **Shape inference** — happens automatically as you connect and edit
5. **Train** — click Train, watch live metrics in the dashboard (F to toggle)
6. **Inspect** — click a node to see detailed visualizations, confusion matrix, loss landscape
7. **Step through** — click Step Through to walk through the forward pass layer by layer
8. **Save/Load** — toolbar icons to export/import graphs and trained weights
9. **Presets** — click the book icon to load a pre-built model

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Tab | Toggle node palette |
| F | Toggle training dashboard |
| ? | Show all shortcuts |
| Escape | Close step-through / modals |
| Ctrl+Z / Ctrl+Shift+Z | Undo / Redo |
| Ctrl+C / Ctrl+V | Copy / Paste nodes |
| Delete | Remove selected nodes |

