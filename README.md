<p align="center">
  <img src="public/nodetorch.png" alt="NodeTorch" width="120" />
</p>
<h1 align="center">NodeTorch</h1>
<p align="center">Node-based visual tool for building, inspecting, and understanding ML models.<br/>Educational and open-source.</p>

---

## Demo

<p align="center">
    <video src="https://github.com/WilsonZheng0327/nodetorch/raw/master/public/Demo.mp4" controls width="720"></video>
</p>

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
- **Save/load** — export graphs as JSON, save/load trained weights to disk
- **Model presets** — one-click load of 11 pre-built architectures
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

## Available Nodes

**Layers**: Conv2d, Conv1d, ConvTranspose2d, Linear, Embedding, LSTM, GRU, RNN, MultiHeadAttention, Flatten, Upsample, Pretrained ResNet-18

**Normalization**: BatchNorm2d/1d, LayerNorm, GroupNorm, InstanceNorm2d

**Activations**: ReLU, Sigmoid, Tanh, Softmax, GELU, LeakyReLU

**Pooling**: MaxPool2d/1d, AvgPool2d, AdaptiveAvgPool2d

**Structural**: Add (residual), Concat, Reshape, Permute, SequencePool, Reparameterize, Subgraph Blocks

**Loss**: CrossEntropy, MSE, VAE Loss, GAN Loss

**GAN**: Noise Input

**Diffusion**: Noise Scheduler

**Optimizers**: SGD, Adam, AdamW — with schedulers (cosine, step, warmup), early stopping, gradient clipping

## Setup

### Frontend

```bash
npm install
npm run dev
```

Opens at http://localhost:5173.

### Backend

Requires Python 3.12+.

```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (pick your CUDA version from https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt

# Run the server
cd backend
python main.py
```

Runs at http://localhost:8000. Auto-detects CUDA/MPS if available.

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add new node types, datasets, and visualizations.
