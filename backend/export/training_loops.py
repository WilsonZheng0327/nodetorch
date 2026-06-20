"""Training-loop and optimizer/scheduler code generation per paradigm."""

from engine.graph_builder import (
    DIFFUSION_SCHEDULER_TYPE, GAN_NOISE_TYPE, OPTIMIZER_NODES,
)
from export.helpers import _get_optimizer_props, _get_optimizer_type
from export.templates import DATASET_CODE

# ─── Training loop templates ─────────────────────────────────────────────────

def _standard_classification_loop(opt_props: dict, opt_type: str, loss_node: dict | None) -> str:
    """Generate standard classification training loop."""
    lr = opt_props.get("lr", 0.001)
    epochs = opt_props.get("epochs", 10)
    scheduler_type = opt_props.get("scheduler", "")

    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append("    criterion = nn.CrossEntropyLoss()")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("        correct = 0")
    lines.append("        total = 0")
    lines.append("")
    lines.append("        for images, labels in train_loader:")
    lines.append("            images, labels = images.to(device), labels.to(device)")
    lines.append("")
    lines.append("            optimizer.zero_grad()")
    lines.append("            outputs = model(images)")
    lines.append("            loss = criterion(outputs, labels)")
    lines.append("            loss.backward()")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("            _, predicted = outputs.max(1)")
    lines.append("            total += labels.size(0)")
    lines.append("            correct += predicted.eq(labels).sum().item()")
    lines.append("")
    if scheduler_type:
        lines.append("        scheduler.step()")
    lines.append("        epoch_loss = running_loss / len(train_loader)")
    lines.append("        epoch_acc = 100.0 * correct / total")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}, Accuracy: {{epoch_acc:.1f}}%")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model")

    return '\n'.join(lines)


def _autoregressive_training_loop(opt_props: dict, opt_type: str) -> str:
    """Generate autoregressive LM training loop (per-token CrossEntropy, perplexity)."""
    import math as _math  # noqa: F401  (used in generated code)
    epochs = opt_props.get("epochs", 10)
    scheduler_type = opt_props.get("scheduler", "")
    grad_clip = opt_props.get("gradClip", 0)
    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    lines.append("import math")
    lines.append("")
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append("    criterion = nn.CrossEntropyLoss()")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("        n_batches = 0")
    lines.append("")
    lines.append("        for inputs, targets in train_loader:")
    lines.append("            inputs, targets = inputs.to(device), targets.to(device)")
    lines.append("")
    lines.append("            optimizer.zero_grad()")
    lines.append("            logits = model(inputs)  # [B, seq_len, vocab_size]")
    lines.append("            B, S, V = logits.shape")
    lines.append("            loss = criterion(logits.reshape(B * S, V), targets.reshape(B * S))")
    lines.append("            loss.backward()")
    if grad_clip and grad_clip > 0:
        lines.append(f"            torch.nn.utils.clip_grad_norm_(model.parameters(), {grad_clip})")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("            n_batches += 1")
    lines.append("")
    if scheduler_type:
        lines.append("        scheduler.step()")
    lines.append("        epoch_loss = running_loss / max(n_batches, 1)")
    lines.append("        ppl = math.exp(min(epoch_loss, 20))")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}, Perplexity: {{ppl:.2f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model")

    return '\n'.join(lines)


def _standard_reconstruction_loop(opt_props: dict, opt_type: str) -> str:
    """Generate MSE reconstruction training loop (autoencoder)."""
    lr = opt_props.get("lr", 0.001)
    epochs = opt_props.get("epochs", 10)
    scheduler_type = opt_props.get("scheduler", "")

    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append("    criterion = nn.MSELoss()")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("")
    lines.append("        for images, _ in train_loader:")
    lines.append("            images = images.to(device)")
    lines.append("")
    lines.append("            optimizer.zero_grad()")
    lines.append("            outputs = model(images)")
    lines.append("            loss = criterion(outputs, images)")
    lines.append("            loss.backward()")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("")
    if scheduler_type:
        lines.append("        scheduler.step()")
    lines.append("        epoch_loss = running_loss / len(train_loader)")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model")

    return '\n'.join(lines)


def _vae_training_loop(opt_props: dict, opt_type: str, loss_props: dict) -> str:
    """Generate VAE training loop."""
    epochs = opt_props.get("epochs", 15)
    beta = loss_props.get("beta", 1.0)
    scheduler_type = opt_props.get("scheduler", "")

    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    lines.append("def vae_loss_fn(reconstruction, original, mean, logvar, beta=1.0):")
    lines.append('    """VAE loss = reconstruction loss + beta * KL divergence."""')
    lines.append("    recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='mean')")
    lines.append("    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())")
    lines.append("    return recon_loss + beta * kl_loss")
    lines.append("")
    lines.append("")
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("")
    lines.append("        for images, _ in train_loader:")
    lines.append("            images = images.to(device)")
    lines.append("")
    lines.append("            optimizer.zero_grad()")
    lines.append("            reconstruction, mean, logvar = model(images)")
    lines.append(f"            loss = vae_loss_fn(reconstruction, images, mean, logvar, beta={beta})")
    lines.append("            loss.backward()")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("")
    if scheduler_type:
        lines.append("        scheduler.step()")
    lines.append("        epoch_loss = running_loss / len(train_loader)")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model")

    return '\n'.join(lines)


def _gan_training_loop(graph_data: dict, modules: dict) -> str:
    """Generate GAN training loop with Generator and Discriminator."""
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}

    # Find noise node for latent dim
    latent_dim = 100
    batch_size = 64
    for n in nodes.values():
        if n["type"] == GAN_NOISE_TYPE:
            latent_dim = n.get("properties", {}).get("latentDim", 100)
            batch_size = n.get("properties", {}).get("batchSize", 64)

    # Find optimizer(s) for lr, epochs
    opt_nodes = [n for n in nodes.values() if n["type"] in OPTIMIZER_NODES]
    # Use first optimizer's settings
    opt_props = opt_nodes[0].get("properties", {}) if opt_nodes else {"lr": 0.0002, "epochs": 100}
    epochs = opt_props.get("epochs", 100)
    lr_g = opt_props.get("lr", 0.0002)
    # Second optimizer (if exists) for discriminator
    lr_d = opt_nodes[1].get("properties", {}).get("lr", 0.0001) if len(opt_nodes) > 1 else lr_g
    beta1 = opt_props.get("beta1", 0.5)
    beta2 = opt_props.get("beta2", 0.999)
    label_smoothing = 0.1
    for n in nodes.values():
        if n["type"] == "ml.loss.gan":
            label_smoothing = n.get("properties", {}).get("labelSmoothing", 0.1)

    lines = []
    lines.append("def train():")
    lines.append("    generator = Generator().to(device)")
    lines.append("    discriminator = Discriminator().to(device)")
    lines.append("")
    lines.append(f"    optimizer_g = torch.optim.Adam(generator.parameters(), lr={lr_g}, betas=({beta1}, {beta2}))")
    lines.append(f"    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr={lr_d}, betas=({beta1}, {beta2}))")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        generator.train()")
    lines.append("        discriminator.train()")
    lines.append("        d_loss_total = 0.0")
    lines.append("        g_loss_total = 0.0")
    lines.append("")
    lines.append("        for real_images, _ in train_loader:")
    lines.append("            real_images = real_images.to(device)")
    lines.append(f"            batch_size = real_images.size(0)")
    lines.append("")
    lines.append("            # --- Train Discriminator ---")
    lines.append("            optimizer_d.zero_grad()")
    lines.append(f"            noise = torch.randn(batch_size, {latent_dim}, device=device)")
    lines.append("            fake_images = generator(noise).detach()")
    lines.append("")
    lines.append("            real_scores = discriminator(real_images)")
    lines.append("            fake_scores = discriminator(fake_images)")
    lines.append("")
    lines.append(f"            real_labels = torch.ones_like(real_scores) * {1.0 - label_smoothing}")
    lines.append("            fake_labels = torch.zeros_like(fake_scores)")
    lines.append("            d_loss_real = nn.functional.binary_cross_entropy_with_logits(real_scores, real_labels)")
    lines.append("            d_loss_fake = nn.functional.binary_cross_entropy_with_logits(fake_scores, fake_labels)")
    lines.append("            d_loss = d_loss_real + d_loss_fake")
    lines.append("            d_loss.backward()")
    lines.append("            optimizer_d.step()")
    lines.append("")
    lines.append("            # --- Train Generator ---")
    lines.append("            optimizer_g.zero_grad()")
    lines.append(f"            noise = torch.randn(batch_size, {latent_dim}, device=device)")
    lines.append("            fake_images = generator(noise)")
    lines.append("            fake_scores = discriminator(fake_images)")
    lines.append("            g_loss = nn.functional.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))")
    lines.append("            g_loss.backward()")
    lines.append("            optimizer_g.step()")
    lines.append("")
    lines.append("            d_loss_total += d_loss.item()")
    lines.append("            g_loss_total += g_loss.item()")
    lines.append("")
    lines.append("        n_batches = len(train_loader)")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — D Loss: {{d_loss_total/n_batches:.4f}}, G Loss: {{g_loss_total/n_batches:.4f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return generator, discriminator")

    return '\n'.join(lines)


def _diffusion_training_loop(graph_data: dict, modules: dict) -> str:
    """Generate diffusion training loop with noise scheduler."""
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}

    # Find scheduler properties
    num_timesteps = 100
    beta_start = 0.0001
    beta_end = 0.02
    schedule_type = "linear"
    for n in nodes.values():
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            props = n.get("properties", {})
            num_timesteps = props.get("numTimesteps", 100)
            beta_start = props.get("betaStart", 0.0001)
            beta_end = props.get("betaEnd", 0.02)
            schedule_type = props.get("scheduleType", "linear")

    # Optimizer props
    opt_props = _get_optimizer_props(nodes)
    opt_type = _get_optimizer_type(nodes)
    epochs = opt_props.get("epochs", 50)
    scheduler_type = opt_props.get("scheduler", "")

    opt_code = _optimizer_code(opt_type, opt_props)

    lines = []
    # Noise scheduler class
    lines.append("class NoiseScheduler:")
    lines.append('    """DDPM noise scheduler for diffusion training."""')
    lines.append("")
    lines.append(f"    def __init__(self, num_timesteps={num_timesteps}, beta_start={beta_start}, beta_end={beta_end}):")
    if schedule_type == "cosine":
        lines.append("        # Cosine schedule")
        lines.append("        steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps")
        lines.append("        alpha_bar = torch.cos((steps + 0.008) / 1.008 * 3.14159265 / 2) ** 2")
        lines.append("        alpha_bar = alpha_bar / alpha_bar[0]")
        lines.append("        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])")
        lines.append("        self.betas = torch.clamp(betas, 0.0001, 0.999).float()")
    else:
        lines.append("        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)")
    lines.append("        self.alphas = 1.0 - self.betas")
    lines.append("        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)")
    lines.append("        self.num_timesteps = num_timesteps")
    lines.append("")
    lines.append("    def add_noise(self, x, noise, t):")
    lines.append('        """Add noise at timestep t: x_t = sqrt(alpha_bar_t) * x + sqrt(1-alpha_bar_t) * noise."""')
    lines.append("        sqrt_alpha = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1).to(x.device)")
    lines.append("        sqrt_one_minus = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1).to(x.device)")
    lines.append("        return sqrt_alpha * x + sqrt_one_minus * noise")
    lines.append("")
    lines.append("    def sample_step(self, model, x_t, t):")
    lines.append('        """One denoising step: predict noise, remove it."""')
    lines.append("        beta = self.betas[t].to(x_t.device)")
    lines.append("        alpha = self.alphas[t].to(x_t.device)")
    lines.append("        alpha_bar = self.alpha_cumprod[t].to(x_t.device)")
    lines.append("")
    lines.append("        # Predict noise")
    lines.append("        t_channel = torch.full((x_t.size(0), 1, x_t.size(2), x_t.size(3)), t / self.num_timesteps, device=x_t.device)")
    lines.append("        model_input = torch.cat([x_t, t_channel], dim=1)")
    lines.append("        predicted_noise = model(model_input)")
    lines.append("")
    lines.append("        # Compute x_{t-1}")
    lines.append("        x_prev = (1 / alpha.sqrt()) * (x_t - (beta / (1 - alpha_bar).sqrt()) * predicted_noise)")
    lines.append("        if t > 0:")
    lines.append("            noise = torch.randn_like(x_t)")
    lines.append("            x_prev = x_prev + beta.sqrt() * noise")
    lines.append("        return x_prev")
    lines.append("")
    lines.append("")
    # Training function
    lines.append("def train():")
    lines.append("    model = Model().to(device)")
    lines.append(f"    scheduler = NoiseScheduler(num_timesteps={num_timesteps})")
    lines.append(f"    optimizer = {opt_code}")
    if scheduler_type:
        lines.append(f"    lr_scheduler = {_scheduler_code(scheduler_type, epochs)}")
    lines.append("    criterion = nn.MSELoss()")
    lines.append("")
    lines.append(f"    for epoch in range({epochs}):")
    lines.append("        model.train()")
    lines.append("        running_loss = 0.0")
    lines.append("")
    lines.append("        for images, _ in train_loader:")
    lines.append("            images = images.to(device)")
    lines.append("            batch_size = images.size(0)")
    lines.append("")
    lines.append("            # Sample random timesteps")
    lines.append(f"            t = torch.randint(0, {num_timesteps}, (batch_size,))")
    lines.append("            noise = torch.randn_like(images)")
    lines.append("            noisy_images = scheduler.add_noise(images, noise, t)")
    lines.append("")
    lines.append("            # Concatenate timestep channel")
    lines.append(f"            t_normalized = t.float() / {num_timesteps}")
    lines.append("            t_channel = t_normalized.view(-1, 1, 1, 1).expand(-1, 1, images.size(2), images.size(3)).to(device)")
    lines.append("            model_input = torch.cat([noisy_images, t_channel], dim=1)")
    lines.append("")
    lines.append("            # Predict and compute loss")
    lines.append("            optimizer.zero_grad()")
    lines.append("            predicted_noise = model(model_input)")
    lines.append("            loss = criterion(predicted_noise, noise)")
    lines.append("            loss.backward()")
    lines.append("            optimizer.step()")
    lines.append("")
    lines.append("            running_loss += loss.item()")
    lines.append("")
    if scheduler_type:
        lines.append("        lr_scheduler.step()")
    lines.append("        epoch_loss = running_loss / len(train_loader)")
    lines.append(f'        print(f"Epoch {{epoch+1}}/{epochs} — Loss: {{epoch_loss:.4f}}")')
    lines.append("")
    lines.append("    print('Training complete!')")
    lines.append("    return model, scheduler")
    lines.append("")
    lines.append("")
    # Sampling function
    lines.append("@torch.no_grad()")
    lines.append("def sample(model, scheduler, num_samples=16):")
    lines.append('    """Generate images by denoising from pure noise."""')
    lines.append("    model.eval()")
    lines.append(f"    x = torch.randn(num_samples, {_get_image_channels_for_diffusion(nodes)}, {_get_image_size_for_diffusion(nodes)}, {_get_image_size_for_diffusion(nodes)}, device=device)")
    lines.append(f"    for t in reversed(range({num_timesteps})):")
    lines.append("        x = scheduler.sample_step(model, x, t)")
    lines.append("    return x.clamp(-1, 1)")

    return '\n'.join(lines)


def _get_image_channels_for_diffusion(nodes: dict) -> int:
    """Get image channels from the data node."""
    for n in nodes.values():
        if n["type"] in DATASET_CODE:
            return DATASET_CODE[n["type"]]["channels"]
    return 1


def _get_image_size_for_diffusion(nodes: dict) -> int:
    """Get image size from the data node."""
    for n in nodes.values():
        if n["type"] in DATASET_CODE:
            return DATASET_CODE[n["type"]]["image_size"]
    return 28


# ─── Optimizer code generation ────────────────────────────────────────────────

def _optimizer_code(opt_type: str, props: dict) -> str:
    """Generate optimizer constructor call."""
    lr = props.get("lr", 0.001)
    if opt_type == "ml.optimizers.sgd":
        momentum = props.get("momentum", 0.9)
        wd = props.get("weightDecay", 0)
        code = f"torch.optim.SGD(model.parameters(), lr={lr}, momentum={momentum}"
        if wd:
            code += f", weight_decay={wd}"
        return code + ")"
    elif opt_type == "ml.optimizers.adamw":
        wd = props.get("weightDecay", 0.01)
        return f"torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay={wd})"
    else:  # adam
        beta1 = props.get("beta1", 0.9)
        beta2 = props.get("beta2", 0.999)
        if beta1 != 0.9 or beta2 != 0.999:
            return f"torch.optim.Adam(model.parameters(), lr={lr}, betas=({beta1}, {beta2}))"
        return f"torch.optim.Adam(model.parameters(), lr={lr})"


def _scheduler_code(scheduler_type: str, epochs: int) -> str:
    """Generate LR scheduler code."""
    if scheduler_type == "cosine":
        return f"torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={epochs})"
    elif scheduler_type == "step":
        return f"torch.optim.lr_scheduler.StepLR(optimizer, step_size={max(1, epochs // 3)}, gamma=0.1)"
    return ""


