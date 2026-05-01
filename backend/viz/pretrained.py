"""Pretrained model visualizations (ResNet-18, etc.)."""

import torch
from graph_builder import _safe_float
from .helpers import feature_maps_data, vector_data, histogram_data, EMPTY_FMAPS


def forward_viz_pretrained_resnet18(node_type, module, input_tensor, output, inputs, out_dict):
    """Pretrained ResNet-18: show input image, model info, output features."""
    transformation: dict = {"type": "pretrained"}

    # Model metadata
    transformation["modelName"] = "ResNet-18"
    transformation["pretrainedOn"] = "ImageNet (1.2M images, 1000 classes)"
    transformation["topAcc"] = "69.8% top-1, 89.1% top-5"
    transformation["totalParams"] = "11.7M"

    if module is not None:
        freeze = True
        trainable = 0
        total = 0
        for p in module.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                freeze = False
        transformation["trainableParams"] = f"{trainable:,}"
        transformation["frozen"] = (trainable == 0)
        mode = getattr(module, 'mode', 'features')
        transformation["mode"] = mode

        # Architecture breakdown
        layers = []
        layers.append({"name": "conv1", "detail": "7×7, 64 filters, stride 2"})
        layers.append({"name": "bn1 + relu", "detail": "BatchNorm + ReLU"})
        layers.append({"name": "maxpool", "detail": "3×3, stride 2"})
        layers.append({"name": "layer1", "detail": "2× BasicBlock (64 filters)"})
        layers.append({"name": "layer2", "detail": "2× BasicBlock (128 filters)"})
        layers.append({"name": "layer3", "detail": "2× BasicBlock (256 filters)"})
        layers.append({"name": "layer4", "detail": "2× BasicBlock (512 filters)"})
        layers.append({"name": "avgpool", "detail": "AdaptiveAvgPool → 1×1"})
        if mode == "logits":
            layers.append({"name": "fc", "detail": "512 → 1000 (ImageNet classes)"})
        else:
            layers.append({"name": "fc", "detail": "removed (feature mode: → 512)"})
        transformation["architecture"] = layers

    # Input
    if input_tensor is not None and isinstance(input_tensor, torch.Tensor) and input_tensor.dim() == 4:
        transformation["inputFmaps"] = feature_maps_data(input_tensor[0])
        transformation["inputShape"] = list(input_tensor.shape)

    # Output
    if output is not None and isinstance(output, torch.Tensor):
        transformation["outputShape"] = list(output.shape)
        if output.dim() == 4:
            transformation["outputFmaps"] = feature_maps_data(output[0])
        elif output.dim() >= 1:
            out_vec = output[0].detach().float().flatten() if output.dim() >= 2 else output.detach().float().flatten()
            transformation["outputVector"] = [_safe_float(float(v)) for v in out_vec[:128].tolist()]
            transformation["outputDim"] = int(out_vec.numel())
            transformation["outputHist"] = histogram_data(output)

    freeze_str = "frozen" if transformation.get("frozen") else "fine-tuning"
    mode_str = "feature extractor (→ 512-dim)" if transformation.get("mode") == "features" else "full model (→ 1000 classes)"
    insight = f"Pretrained ResNet-18 ({freeze_str}, {mode_str}) — trained on ImageNet with 69.8% accuracy"
    return {"transformation": transformation, "insight": insight}
