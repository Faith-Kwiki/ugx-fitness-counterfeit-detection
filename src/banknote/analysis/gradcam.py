"""Generate a Grad-CAM heatmap for one image and a completed run."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.banknote.models.research_models import build_research_model
from src.banknote.preprocessing.prepare_streams import prepare_banknote_streams


LABELS = {0: "counterfeit", 1: "genuine"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--signal", choices=("rgb", "edges", "texture"), default="rgb")
    parser.add_argument("--class-index", type=int, choices=(0, 1), default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def find_target_layer(model_name, model, signal):
    if model_name in {"resnet18_frozen", "resnet18_finetune", "hybrid_concat", "hybrid_attention"}:
        if signal != "rgb":
            raise ValueError("This model's ResNet feature map is the RGB Grad-CAM target.")
        return model.backbone.layer4[-1]
    if model_name == "rgb_cnn":
        if signal != "rgb":
            raise ValueError("RGB CNN Grad-CAM only supports the rgb signal.")
        return model.network.features[-3]
    if model_name == "multisignal_concat":
        branch_name = {"rgb": "rgb_branch", "edges": "edge_branch", "texture": "texture_branch"}[signal]
        return getattr(model, branch_name).features[-4]
    if model_name == "signal_ablation":
        if signal not in model.branches:
            raise ValueError(f"Run does not include the {signal} signal.")
        return model.branches[signal].features[-4]
    raise ValueError(f"Grad-CAM is not configured for model {model_name}.")


def main():
    args = parse_args()
    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best_checkpoint.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    model_name = config.get("model")
    if not model_name:
        model_name = (
            "multisignal_concat"
            if config.get("architecture") == "MultiSignalCNN"
            else "signal_ablation"
        )
    signals = tuple(config.get("signals", ("rgb", "edges", "texture")))
    model = build_research_model(model_name, signals=signals, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_size = int(config.get("image_size", 224))
    streams = prepare_banknote_streams(str(args.image), target_size=target_size)
    tensors = {
        "rgb": torch.from_numpy(
            np.ascontiguousarray(streams["rgb"].transpose(2, 0, 1))
        ).float().unsqueeze(0).div_(255.0),
        "edges": torch.from_numpy(
            np.ascontiguousarray(streams["edges"][None, :, :])
        ).float().unsqueeze(0).div_(255.0),
        "texture": torch.from_numpy(
            np.ascontiguousarray(streams["texture"][None, :, :])
        ).float().unsqueeze(0).div_(255.0),
    }
    for tensor in tensors.values():
        tensor.requires_grad_(True)
    target_layer = find_target_layer(model_name, model, args.signal)
    activations = {}
    gradients = {}

    def capture_activation(module, inputs, output):
        activations["value"] = output
        output.register_hook(
            lambda gradient: gradients.__setitem__("value", gradient)
        )

    forward_handle = target_layer.register_forward_hook(capture_activation)
    model.zero_grad(set_to_none=True)
    logits = model(tensors["rgb"], tensors["edges"], tensors["texture"])
    class_index = int(logits.argmax(dim=1).item()) if args.class_index is None else args.class_index
    logits[0, class_index].backward()
    forward_handle.remove()

    feature_map = activations["value"]
    gradient = gradients["value"]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    if args.signal == "rgb":
        background = streams["rgb"]
    else:
        background = np.repeat(streams[args.signal][:, :, None], 3, axis=2)
    heatmap_bgr = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(
        0.55 * background.astype(np.float32)
        + 0.45 * heatmap.astype(np.float32)
    )
    output_path = args.output or (
        run_dir / f"gradcam_{args.signal}_{args.image.stem}_{LABELS[class_index]}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    result = {
        "run_id": config.get("run_id"),
        "model": model_name,
        "image": str(args.image),
        "signal": args.signal,
        "predicted_class": LABELS[int(logits.argmax(dim=1).item())],
        "explained_class": LABELS[class_index],
        "explained_class_probability": float(torch.softmax(logits.detach(), dim=1)[0, class_index]),
        "heatmap": str(output_path),
        "interpretation_note": "Grad-CAM highlights regions associated with this prediction; it does not prove causal use of a security feature.",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
