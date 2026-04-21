# node_builders.py — Per-node-type builder functions.
#
# Each builder takes (properties, input_shapes) and returns a torch.nn.Module.
# input_shapes is a dict of port_id → shape list, derived from upstream tensors.
# This is how nodes that need input dimensions (Conv2d needs in_channels, Linear needs
# in_features) get them — the shapes are computed from actual tensor data during the
# initial forward pass in build_and_run_graph().
#
# Wrapper modules (LSTMWrapper, GRUWrapper, MHAWrapper, ConcatModule, etc.) exist
# because some PyTorch modules don't match our calling convention:
#   - LSTM/GRU return tuples → wrappers return dicts for multi-output routing
#   - MHA takes (Q,K,V) → wrapper accepts named kwargs
#   - Concat needs sorted inputs → wrapper sorts by key prefix
#   - F.scaled_dot_product_attention is functional → AttentionModule wraps it
#
# NODE_BUILDERS registry maps node type strings → builder functions.
# graph_builder.py looks up builders from this registry.

import torch
import torch.nn as nn


# --- Layers (no wrapper needed — single tensor in, single tensor out) ---

def build_conv2d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    in_channels = in_shape[1] if in_shape else 1
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=props["outChannels"],
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


def build_conv1d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    in_channels = in_shape[1] if in_shape else 1
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=props["outChannels"],
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


def build_linear(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    in_features = in_shape[-1] if in_shape else 1
    return nn.Linear(
        in_features=in_features,
        out_features=props["outFeatures"],
    )


def build_flatten(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Flatten()


def build_maxpool2d(props: dict, input_shapes: dict) -> nn.Module:
    return nn.MaxPool2d(
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


def build_avgpool2d(props: dict, input_shapes: dict) -> nn.Module:
    return nn.AvgPool2d(
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


def build_adaptive_avgpool2d(props: dict, input_shapes: dict) -> nn.Module:
    return nn.AdaptiveAvgPool2d((props["outputH"], props["outputW"]))


def build_batchnorm2d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    num_features = in_shape[1] if in_shape else 1
    return nn.BatchNorm2d(num_features=num_features)


def build_batchnorm1d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    num_features = in_shape[1] if in_shape else 1
    return nn.BatchNorm1d(num_features=num_features)


def build_dropout(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Dropout(p=props["p"])


def build_layernorm(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    n = props.get("numLastDims", 1)
    normalized_shape = in_shape[-n:] if in_shape else [1]
    return nn.LayerNorm(normalized_shape)


def build_embedding(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Embedding(
        num_embeddings=props["numEmbeddings"],
        embedding_dim=props["embeddingDim"],
    )


# --- Layers with wrappers (multi-output or non-standard calling convention) ---

class LSTMWrapper(nn.Module):
    """Why wrapper: nn.LSTM returns a tuple (output, (hidden, cell)). Our execution
    loop expects either a single tensor or a dict. This unpacks the tuple into a dict
    so each output can be routed to different downstream nodes via port IDs."""
    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        self.lstm = lstm

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, (hidden, cell) = self.lstm(x)
        return {"out": output, "hidden": hidden, "cell": cell}


def build_lstm(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    input_size = in_shape[-1] if in_shape else 1
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=props["hiddenSize"],
        num_layers=props.get("numLayers", 1),
        batch_first=True,
        bidirectional=props.get("bidirectional", False),
        dropout=props.get("dropout", 0) if props.get("numLayers", 1) > 1 else 0,
    )
    return LSTMWrapper(lstm)


class GRUWrapper(nn.Module):
    """Why wrapper: nn.GRU returns a tuple (output, hidden). Same reason as LSTMWrapper —
    unpacks the tuple into a dict for multi-output port routing."""
    def __init__(self, gru: nn.GRU):
        super().__init__()
        self.gru = gru

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, hidden = self.gru(x)
        return {"out": output, "hidden": hidden}


def build_gru(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    input_size = in_shape[-1] if in_shape else 1
    gru = nn.GRU(
        input_size=input_size,
        hidden_size=props["hiddenSize"],
        num_layers=props.get("numLayers", 1),
        batch_first=True,
        bidirectional=props.get("bidirectional", False),
        dropout=props.get("dropout", 0) if props.get("numLayers", 1) > 1 else 0,
    )
    return GRUWrapper(gru)


class MHAWrapper(nn.Module):
    """Why wrapper: nn.MultiheadAttention returns (attn_output, attn_weights) — we only
    want the output. Also, our multi-input path calls module(**kwargs) with named args
    (query, key, value), which MHA accepts but also returns the unwanted weights tuple."""
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.mha(query, key, value)
        return output


def build_multihead_attention(props: dict, input_shapes: dict) -> nn.Module:
    mha = nn.MultiheadAttention(
        embed_dim=props["embedDim"],
        num_heads=props["numHeads"],
        dropout=props.get("dropout", 0.0),
        batch_first=True,
    )
    return MHAWrapper(mha)


class AttentionModule(nn.Module):
    """Why wrapper: F.scaled_dot_product_attention is a function, not an nn.Module.
    Can't be registered in nn.ModuleDict, has no .parameters(). This wraps it
    so it behaves like any other module in the execution loop."""
    def __init__(self, dropout: float = 0.0, is_causal: bool = False):
        super().__init__()
        self.dropout = dropout
        self.is_causal = is_causal

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )


def build_attention(props: dict, input_shapes: dict) -> nn.Module:
    return AttentionModule(
        dropout=props.get("dropout", 0.0),
        is_causal=props.get("causalMask", False),
    )


# --- Activations (no wrapper needed — all are single tensor in/out) ---

def build_relu(props: dict, input_shapes: dict) -> nn.Module:
    return nn.ReLU()


def build_sigmoid(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Sigmoid()


def build_softmax(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Softmax(dim=props.get("dim", -1))


def build_gelu(props: dict, input_shapes: dict) -> nn.Module:
    return nn.GELU()


def build_tanh(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Tanh()


def build_leaky_relu(props: dict, input_shapes: dict) -> nn.Module:
    return nn.LeakyReLU(negative_slope=props.get("negativeSlope", 0.01))


# --- Loss ---

class CrossEntropyLossWrapper(nn.Module):
    """CrossEntropy that auto-reshapes for sequence models.
    Standard: predictions [B, C], labels [B].
    Sequence: predictions [B, seq_len, C], labels [B, seq_len].
    Reshapes the latter to [B*seq_len, C] and [B*seq_len] automatically."""
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if predictions.dim() == 3:
            # [B, seq_len, vocab_size] → [B*seq_len, vocab_size]
            B, S, C = predictions.shape
            predictions = predictions.reshape(B * S, C)
            labels = labels.reshape(B * S)
        return self.loss(predictions, labels)


def build_cross_entropy_loss(props: dict, input_shapes: dict) -> nn.Module:
    return CrossEntropyLossWrapper()


def build_mse_loss(props: dict, input_shapes: dict) -> nn.Module:
    return nn.MSELoss()


class GANLossModule(nn.Module):
    """GAN discriminator loss: BCE with logits on real and fake scores.
    real_labels are smoothed from 1.0 to (1-smoothing) to stabilize training."""
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
        real_labels = torch.ones_like(real_scores) * (1.0 - self.label_smoothing)
        fake_labels = torch.zeros_like(fake_scores)
        d_loss_real = nn.functional.binary_cross_entropy_with_logits(real_scores, real_labels)
        d_loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_scores, fake_labels)
        return d_loss_real + d_loss_fake


def build_gan_loss(props: dict, input_shapes: dict) -> nn.Module:
    return GANLossModule(label_smoothing=props.get("labelSmoothing", 0.1))


class VAELossModule(nn.Module):
    """VAE loss: reconstruction (MSE) + beta * KL divergence.
    Takes 4 named inputs: reconstruction, original, mean, logvar."""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, reconstruction: torch.Tensor, original: torch.Tensor,
                mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss


def build_vae_loss(props: dict, input_shapes: dict) -> nn.Module:
    return VAELossModule(beta=props.get("beta", 1.0))


# --- Structural (all need wrappers — none are native nn.Modules) ---

class NoiseSchedulerModule(nn.Module):
    """Stores the diffusion noise schedule (beta, alpha, alpha_bar).
    During training, the loop uses add_noise() to corrupt clean images.
    forward() is a passthrough for shape inference."""
    def __init__(self, num_timesteps=100, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        super().__init__()
        if schedule_type == 'cosine':
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * 3.14159265 / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, 0.0001, 0.999).float()
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)

        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.num_timesteps = num_timesteps

    def add_noise(self, x, noise, t):
        """Add noise at timestep t: x_t = sqrt(alpha_bar_t) * x + sqrt(1-alpha_bar_t) * noise"""
        sqrt_alpha = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha * x + sqrt_one_minus * noise

    def forward(self, images):
        """Passthrough for shape inference — actual noising happens in training loop."""
        return images


def build_noise_scheduler(props: dict, input_shapes: dict) -> nn.Module:
    return NoiseSchedulerModule(
        num_timesteps=props.get("numTimesteps", 100),
        beta_start=props.get("betaStart", 0.0001),
        beta_end=props.get("betaEnd", 0.02),
        schedule_type=props.get("scheduleType", "linear"),
    )


class TimestepEmbedModule(nn.Module):
    """Sinusoidal timestep embedding (like positional encoding in transformers)."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        """t: integer timestep tensor [B]. Returns [B, embed_dim]."""
        half = self.embed_dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device).float() * (2.0 * 3.14159265 / half))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def build_timestep_embed(props: dict, input_shapes: dict) -> nn.Module:
    return TimestepEmbedModule(embed_dim=props.get("embedDim", 128))


class NoiseInputModule(nn.Module):
    """Placeholder module for GAN noise input. During training, the GAN loop
    injects actual noise tensors. The forward() here returns a dummy sample
    so downstream nodes can determine their input shapes during build."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self) -> torch.Tensor:
        return torch.randn(1, self.latent_dim)


def build_noise_input(props: dict, input_shapes: dict) -> nn.Module:
    return NoiseInputModule(latent_dim=props.get("latentDim", 100))


class ReparameterizeModule(nn.Module):
    """VAE reparameterization trick: z = mean + exp(0.5 * logvar) * noise.
    Allows gradients to flow through the sampling step."""
    def forward(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


def build_reparameterize(props: dict, input_shapes: dict) -> nn.Module:
    return ReparameterizeModule()


class AddModule(nn.Module):
    """Why wrapper: element-wise addition is just `a + b` — not a PyTorch module.
    Needs to be a module so it can be stored in nn.ModuleDict and called via module(**kwargs)."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def build_add(props: dict, input_shapes: dict) -> nn.Module:
    return AddModule()


class ConcatModule(nn.Module):
    """Why wrapper: torch.cat is a function, not a module. Also needs to accept **kwargs
    (in_0, in_1, ...) from our multi-input calling convention and sort them by key."""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        tensors = [v for k, v in sorted(inputs.items()) if k.startswith('in_')]
        return torch.cat(tensors, dim=self.dim)


def build_concat(props: dict, input_shapes: dict) -> nn.Module:
    return ConcatModule(dim=props.get("dim", 1))


class ReshapeModule(nn.Module):
    """Why wrapper: torch.reshape is a tensor method, not a module. Needs to store
    the target shape as state so it can be called as module(x) in the execution loop."""
    def __init__(self, target_shape: list[int]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.target_shape)


def build_reshape(props: dict, input_shapes: dict) -> nn.Module:
    target_str = props.get("targetShape", "-1")
    in_shape = input_shapes.get("in", [1])
    target = [int(s.strip()) for s in target_str.split(",")]
    # Resolve 0s (keep original dimension) but leave -1 for torch.reshape to handle
    # at runtime — resolving -1 at build time bakes in the batch size, which breaks
    # when the actual batch size differs (e.g., last batch or different batch size).
    resolved = [in_shape[i] if v == 0 and i < len(in_shape) else v for i, v in enumerate(target)]
    return ReshapeModule(resolved)


class PermuteModule(nn.Module):
    """Why wrapper: torch.permute is a tensor method, not a module. Same as ReshapeModule —
    stores the dimension order as state."""
    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


def build_permute(props: dict, input_shapes: dict) -> nn.Module:
    dims_str = props.get("dims", "0, 2, 1")
    dims = [int(s.strip()) for s in dims_str.split(",")]
    return PermuteModule(dims)


class SequencePoolModule(nn.Module):
    """Why wrapper: sequence pooling (last/mean/max over dim=1) is a tensor op, not a module."""
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'last':
            return x[:, -1, :]
        elif self.mode == 'mean':
            return x.mean(dim=1)
        elif self.mode == 'max':
            return x.max(dim=1).values
        return x[:, -1, :]


def build_sequence_pool(props: dict, input_shapes: dict) -> nn.Module:
    return SequencePoolModule(props.get("mode", "last"))


# --- New layer builders ---

def build_conv_transpose2d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    in_channels = in_shape[1] if in_shape else 1
    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=props["outChannels"],
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
        output_padding=props.get("outputPadding", 0),
    )


def build_upsample(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Upsample(
        scale_factor=props["scaleFactor"],
        mode=props.get("mode", "nearest"),
    )


def build_dropout2d(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Dropout2d(p=props["p"])


def build_groupnorm(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    num_channels = in_shape[1] if in_shape else 1
    return nn.GroupNorm(num_groups=props["numGroups"], num_channels=num_channels)


def build_instancenorm2d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    num_features = in_shape[1] if in_shape else 1
    return nn.InstanceNorm2d(num_features=num_features, affine=True)


def build_maxpool1d(props: dict, input_shapes: dict) -> nn.Module:
    return nn.MaxPool1d(
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


# Why wrapper: nn.RNN returns (output, hidden) tuple. We need a dict for multi-output.
class RNNWrapper(nn.Module):
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self, x):
        out, hidden = self.rnn(x)
        return {"out": out, "hidden": hidden}


def build_rnn(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    input_size = in_shape[-1] if in_shape else 1
    return RNNWrapper(nn.RNN(
        input_size=input_size,
        hidden_size=props["hiddenSize"],
        num_layers=props["numLayers"],
        batch_first=True,
        bidirectional=props.get("bidirectional", False),
        nonlinearity=props.get("nonlinearity", "tanh"),
    ))


# --- Pretrained models ---

# Why wrapper: torchvision's ResNet expects 224×224 input; we auto-resize. Also,
# "features" mode swaps the final FC for Identity so output is [B, 512] pre-classifier.
class PretrainedResNet18Wrapper(nn.Module):
    def __init__(self, mode: str = "features", freeze: bool = True):
        super().__init__()
        import torchvision.models as _models
        self.mode = mode
        self.model = _models.resnet18(weights=_models.ResNet18_Weights.DEFAULT)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        if mode == "features":
            # Replace final FC with identity — output becomes [B, 512]
            self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Auto-resize to 224×224 (bilinear)
        if x.shape[-2] != 224 or x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False,
            )
        return self.model(x)


def build_pretrained_resnet18(props: dict, input_shapes: dict) -> nn.Module:
    return PretrainedResNet18Wrapper(
        mode=props.get("mode", "features"),
        freeze=props.get("freeze", True),
    )


# --- Registry ---

NODE_BUILDERS: dict[str, callable] = {
    # Layers (no wrapper)
    "ml.layers.conv2d": build_conv2d,
    "ml.layers.conv1d": build_conv1d,
    "ml.layers.conv_transpose2d": build_conv_transpose2d,
    "ml.layers.linear": build_linear,
    "ml.layers.flatten": build_flatten,
    "ml.layers.maxpool2d": build_maxpool2d,
    "ml.layers.maxpool1d": build_maxpool1d,
    "ml.layers.avgpool2d": build_avgpool2d,
    "ml.layers.adaptive_avgpool2d": build_adaptive_avgpool2d,
    "ml.layers.batchnorm2d": build_batchnorm2d,
    "ml.layers.batchnorm1d": build_batchnorm1d,
    "ml.layers.groupnorm": build_groupnorm,
    "ml.layers.instancenorm2d": build_instancenorm2d,
    "ml.layers.dropout": build_dropout,
    "ml.layers.dropout2d": build_dropout2d,
    "ml.layers.layernorm": build_layernorm,
    "ml.layers.embedding": build_embedding,
    "ml.layers.upsample": build_upsample,
    "ml.layers.pretrained_resnet18": build_pretrained_resnet18,
    # Layers (with wrapper)
    "ml.layers.lstm": build_lstm,
    "ml.layers.gru": build_gru,
    "ml.layers.rnn": build_rnn,
    "ml.layers.multihead_attention": build_multihead_attention,
    "ml.layers.attention": build_attention,
    # Activations (no wrapper)
    "ml.activations.relu": build_relu,
    "ml.activations.sigmoid": build_sigmoid,
    "ml.activations.softmax": build_softmax,
    "ml.activations.gelu": build_gelu,
    "ml.activations.tanh": build_tanh,
    "ml.activations.leaky_relu": build_leaky_relu,
    # Loss (no wrapper)
    "ml.loss.cross_entropy": build_cross_entropy_loss,
    "ml.loss.mse": build_mse_loss,
    "ml.loss.vae": build_vae_loss,
    "ml.loss.gan": build_gan_loss,
    # GAN
    "ml.gan.noise_input": build_noise_input,
    # Diffusion
    "ml.diffusion.noise_scheduler": build_noise_scheduler,
    "ml.diffusion.timestep_embed": build_timestep_embed,
    # Structural (all wrapped)
    "ml.structural.reparameterize": build_reparameterize,
    "ml.structural.add": build_add,
    "ml.structural.concat": build_concat,
    "ml.structural.reshape": build_reshape,
    "ml.structural.permute": build_permute,
    "ml.structural.sequence_pool": build_sequence_pool,
}
