from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

from treeco.utils.groundingdino_box_cropping import load_image
import cv2
import numpy as np
import torch
from PIL import Image

from treeco.image_models.download import get_model_path


ImageLike = Union[str, Path, Image.Image, np.ndarray]


def get_default_depth_anything_repo() -> Path:
    env_path = os.getenv("DEPTH_ANYTHING_V2_REPO")

    if env_path:
        return Path(env_path).expanduser().resolve()

    return get_model_path("depth_anything_repo")


def load_depth_anything_v2(
    repo_path: Union[str, Path, None] = None,
    ckpt_path: Union[str, Path, None] = None,
    encoder: str = "vitb",
    features: int = 128,
    out_channels: list[int] = [96, 192, 384, 768],
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if repo_path is None:
        repo_path = get_default_depth_anything_repo()

    if ckpt_path is None:
        ckpt_path = get_model_path("depth_anything")

    repo_path = Path(repo_path)
    ckpt_path = Path(ckpt_path)

    if not repo_path.exists():
        raise FileNotFoundError(
            f"Depth Anything repo not found:\n"
            f"  {repo_path}\n\n"
            f"Run:\n"
            f"  treeco-download-models"
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Depth Anything checkpoint not found:\n"
            f"  {ckpt_path}\n\n"
            f"Run:\n"
            f"  treeco-download-models"
        )

    if str(repo_path) not in sys.path:
        sys.path.append(str(repo_path))

    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder=encoder,
        features=features,
        out_channels=out_channels,
    )

    state = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(state)
    model = model.to(device).eval()

    print(f"Loaded Depth Anything V2 model from {ckpt_path} on {device}")

    return model, device


def infer_depth(
    image_or_path: ImageLike,
    model,
    device: str,
    input_size: int = 518,
) -> np.ndarray:
    image = load_image(image_or_path)
    img = np.array(image)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (input_size, input_size))

    x = torch.from_numpy(img_resized).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(x)

    depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
    depth = cv2.resize(depth, (w, h))

    return depth


def normalize_depth(depth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    return (depth - depth_min) / (depth_max - depth_min + eps)


def depth_to_pil(depth: np.ndarray) -> Image.Image:
    depth_norm = normalize_depth(depth)
    return Image.fromarray((depth_norm * 255).astype(np.uint8)).convert("L")