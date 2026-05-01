"""Diffusion-specific visualizations (NoiseScheduler, TimestepEmbed)."""

import torch
from graph_builder import _safe_float
from .helpers import feature_maps_data, EMPTY_FMAPS


def forward_viz_noise_scheduler(node_type, module, input_tensor, output, inputs, out_dict):
    """Noise scheduler: show clean image + noise = noisy image, plus timestep channel."""
    transformation: dict = {"type": "noise_scheduler"}

    # Clean input image
    clean = inputs.get("images")
    if clean is not None and isinstance(clean, torch.Tensor) and clean.dim() == 4:
        transformation["cleanFmaps"] = feature_maps_data(clean[0])
        transformation["cleanShape"] = list(clean.shape)

    # Noise that was added
    noise = out_dict.get("noise") if isinstance(out_dict, dict) else None
    if noise is not None and isinstance(noise, torch.Tensor) and noise.dim() == 4:
        n = noise[0].detach().float()
        nmin, nmax = float(n.min()), float(n.max())
        rng = nmax - nmin if nmax != nmin else 1.0
        normalized = ((n - nmin) / rng * 255).clamp(0, 255)
        C = normalized.shape[0]
        maps = []
        for c in range(min(C, 1)):
            maps.append(normalized[c].byte().tolist())
        transformation["noiseFmaps"] = {
            "maps": maps, "channels": C, "showing": len(maps),
            "height": int(n.shape[1]), "width": int(n.shape[2]),
        }

    # Noisy output
    if output is not None and isinstance(output, torch.Tensor) and output.dim() == 4:
        transformation["noisyFmaps"] = feature_maps_data(output[0])
        transformation["noisyShape"] = list(output.shape)

    # Timestep channel info
    timestep = out_dict.get("timestep") if isinstance(out_dict, dict) else None
    if timestep is not None and isinstance(timestep, torch.Tensor):
        t_val = float(timestep[0, 0, 0, 0])
        num_timesteps = getattr(module, 'num_timesteps', 100) if module is not None else 100
        t_int = int(round(t_val * num_timesteps))
        transformation["timestep"] = t_int
        transformation["numTimesteps"] = num_timesteps
        transformation["tNormalized"] = _safe_float(t_val)
        transformation["timestepShape"] = list(timestep.shape)

        if module is not None and hasattr(module, 'alpha_cumprod'):
            t_idx = min(t_int, num_timesteps - 1)
            alpha_bar = float(module.alpha_cumprod[t_idx])
            transformation["signalRatio"] = _safe_float(alpha_bar)
            transformation["noiseRatio"] = _safe_float(1 - alpha_bar)

    # Show what the concat will produce
    if output is not None and isinstance(output, torch.Tensor) and timestep is not None and isinstance(timestep, torch.Tensor):
        out_ch = output.shape[1]
        t_ch = timestep.shape[1]
        transformation["concatResult"] = f"[1, {out_ch + t_ch}, {output.shape[2]}, {output.shape[3]}]"
        transformation["concatExplain"] = f"{out_ch} image channels + {t_ch} timestep channel = {out_ch + t_ch} total"

    insight = None
    if "timestep" in transformation:
        t = transformation["timestep"]
        T = transformation["numTimesteps"]
        sig = transformation.get("signalRatio")
        if sig is not None:
            insight = f"Timestep {t}/{T} — {sig:.0%} signal, {1-sig:.0%} noise. Outputs: noisy image + noise target + timestep channel"
        else:
            insight = f"Added noise at timestep {t}/{T}. Outputs: noisy image + noise target + timestep channel"
    else:
        insight = "Noise scheduler — adds noise to clean images for training"

    return {"transformation": transformation, "insight": insight}
