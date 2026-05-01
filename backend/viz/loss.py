"""Loss node visualizations (CrossEntropy, MSE, VAE, etc.)."""

import math
import torch
from graph_builder import _safe_float
from .helpers import TOP_K_PROBS, feature_maps_data


def forward_viz_loss(node_type, module, input_tensor, output, inputs, out_dict):
    """Detect cross-entropy and show prediction breakdown, else scalar."""
    if node_type == "ml.loss.cross_entropy":
        preds = inputs.get("predictions")
        labels = inputs.get("labels")
        if (preds is not None and isinstance(preds, torch.Tensor) and preds.dim() >= 2
                and labels is not None and isinstance(labels, torch.Tensor)):
            logits = preds[0].detach().float().flatten()
            probs = torch.softmax(logits, dim=0)
            true_label = int(labels[0].item())
            true_prob = _safe_float(float(probs[true_label]))
            loss_val = _safe_float(-math.log(max(float(probs[true_label]), 1e-12)))
            k = min(TOP_K_PROBS, logits.numel())
            top_vals, top_idx = torch.topk(probs, k)
            transformation: dict = {
                "type": "cross_entropy",
                "logits": [_safe_float(float(v)) for v in logits.tolist()],
                "probabilities": [_safe_float(float(v)) for v in probs.tolist()],
                "trueLabel": true_label,
                "trueLabelProb": true_prob,
                "loss": loss_val,
                "topK": [{"index": int(i), "value": _safe_float(float(v))}
                         for i, v in zip(top_idx.tolist(), top_vals.tolist())],
            }
            return {
                "transformation": transformation,
                "insight": f"Loss = -log(P[class {true_label}]) = -log({true_prob:.4f}) = {loss_val:.4f}",
            }

    # MSE loss: show predictions vs targets side by side
    if node_type == "ml.loss.mse":
        preds = inputs.get("predictions")
        targets = inputs.get("labels")
        transformation_mse: dict = {"type": "mse_loss"}

        if preds is not None and isinstance(preds, torch.Tensor):
            if preds.dim() == 4:
                transformation_mse["predsFmaps"] = feature_maps_data(preds[0])
            transformation_mse["predsShape"] = list(preds.shape)

        if targets is not None and isinstance(targets, torch.Tensor):
            if targets.dim() == 4:
                transformation_mse["targetsFmaps"] = feature_maps_data(targets[0])
            transformation_mse["targetsShape"] = list(targets.shape)

        if output is not None and isinstance(output, torch.Tensor):
            transformation_mse["loss"] = _safe_float(float(output.detach().flatten()[0]))

        # Detailed computation breakdown
        if (preds is not None and isinstance(preds, torch.Tensor)
                and targets is not None and isinstance(targets, torch.Tensor)):
            diff = (preds - targets).detach().float()
            sq_diff = diff.pow(2)
            n_elements = int(diff.numel())
            sum_sq = _safe_float(float(sq_diff.sum()))
            mean_sq = _safe_float(float(sq_diff.mean()))
            max_err = _safe_float(float(diff.abs().max()))
            mean_abs = _safe_float(float(diff.abs().mean()))
            transformation_mse["numElements"] = n_elements
            transformation_mse["sumSquared"] = sum_sq
            transformation_mse["meanSquared"] = mean_sq
            transformation_mse["maxAbsError"] = max_err
            transformation_mse["meanAbsError"] = mean_abs

            # Per-pixel error map (if images)
            if preds.dim() == 4 and targets.dim() == 4:
                spatial_err = diff[0].abs().mean(dim=0)
                dmin, dmax = float(spatial_err.min()), float(spatial_err.max())
                drng = dmax - dmin if dmax > dmin else 1.0
                error_map = ((spatial_err - dmin) / drng * 255).clamp(0, 255).byte()
                transformation_mse["errorMap"] = error_map.tolist()
                transformation_mse["errorH"] = int(error_map.shape[0])
                transformation_mse["errorW"] = int(error_map.shape[1])

        insight = None
        if output is not None and isinstance(output, torch.Tensor):
            val = _safe_float(float(output.detach().flatten()[0]))
            insight = f"MSE Loss = {val:.4f}"
        return {"transformation": transformation_mse, "insight": insight}

    # GAN loss: show real vs fake scores and loss breakdown
    if node_type == "ml.loss.gan":
        real_scores = inputs.get("real_scores")
        fake_scores = inputs.get("fake_scores")

        transformation_gan: dict = {"type": "gan_loss"}

        if real_scores is not None and isinstance(real_scores, torch.Tensor):
            rs = real_scores.detach().float().flatten()
            transformation_gan["realScore"] = _safe_float(float(rs.mean()))
            transformation_gan["realProb"] = _safe_float(float(torch.sigmoid(rs).mean()))

        if fake_scores is not None and isinstance(fake_scores, torch.Tensor):
            fs = fake_scores.detach().float().flatten()
            transformation_gan["fakeScore"] = _safe_float(float(fs.mean()))
            transformation_gan["fakeProb"] = _safe_float(float(torch.sigmoid(fs).mean()))

        if output is not None and isinstance(output, torch.Tensor):
            transformation_gan["totalLoss"] = _safe_float(float(output.detach().flatten()[0]))

        # Individual loss components
        if real_scores is not None and isinstance(real_scores, torch.Tensor):
            smoothing = 0.1
            if module is not None and hasattr(module, 'label_smoothing'):
                smoothing = float(module.label_smoothing)
            real_labels = torch.ones_like(real_scores) * (1.0 - smoothing)
            d_loss_real = float(torch.nn.functional.binary_cross_entropy_with_logits(
                real_scores, real_labels).detach())
            transformation_gan["dLossReal"] = _safe_float(d_loss_real)

        if fake_scores is not None and isinstance(fake_scores, torch.Tensor):
            fake_labels = torch.zeros_like(fake_scores)
            d_loss_fake = float(torch.nn.functional.binary_cross_entropy_with_logits(
                fake_scores, fake_labels).detach())
            transformation_gan["dLossFake"] = _safe_float(d_loss_fake)

        return {"transformation": transformation_gan, "insight": "How well does the discriminator distinguish real training images from generator fakes?"}

    # VAE loss: reconstruction + KL divergence
    if node_type == "ml.loss.vae":
        recon = inputs.get("reconstruction")
        original = inputs.get("original")
        mean = inputs.get("mean")
        logvar = inputs.get("logvar")

        transformation_vae: dict = {"type": "vae_loss"}

        # Show original vs reconstruction images
        if original is not None and isinstance(original, torch.Tensor) and original.dim() == 4:
            transformation_vae["originalFmaps"] = feature_maps_data(original[0])
        if recon is not None and isinstance(recon, torch.Tensor) and recon.dim() == 4:
            transformation_vae["reconFmaps"] = feature_maps_data(recon[0])

        # Compute individual loss components — must match VAELossModule exactly
        # VAELossModule uses reduction='mean' for both components
        if (recon is not None and isinstance(recon, torch.Tensor)
                and original is not None and isinstance(original, torch.Tensor)):
            recon_loss = float(torch.nn.functional.mse_loss(recon, original, reduction='mean').detach())
            transformation_vae["reconLoss"] = _safe_float(recon_loss)

        if (mean is not None and isinstance(mean, torch.Tensor)
                and logvar is not None and isinstance(logvar, torch.Tensor)):
            kl = float((-0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())).detach())
            transformation_vae["klLoss"] = _safe_float(kl)

        # Get beta from module if available
        beta = 1.0
        if module is not None and hasattr(module, 'beta'):
            beta = float(module.beta)
        transformation_vae["beta"] = _safe_float(beta)

        if output is not None and isinstance(output, torch.Tensor):
            transformation_vae["totalLoss"] = _safe_float(float(output.detach().flatten()[0]))

        return {
            "transformation": transformation_vae,
            "insight": "VAE Loss = Reconstruction (MSE) + KL Divergence",
        }

    # Fallback: plain scalar loss
    transformation_scalar: dict = {"type": "loss", "value": 0.0}
    insight = None
    if output is not None and isinstance(output, torch.Tensor):
        val = float(output.detach().flatten()[0])
        transformation_scalar["value"] = _safe_float(val)
        insight = f"Loss value: {val:.4f}"
    return {"transformation": transformation_scalar, "insight": insight}
