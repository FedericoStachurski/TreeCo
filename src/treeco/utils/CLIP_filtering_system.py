from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from treeco.image_models.download import get_model_path


ImageLike = Union[str, Path, Image.Image]


DEFAULT_TREE_PROMPTS = [
    "tree",
    "tree trunk",
    "tree canopy",
    "branches and leaves",
    "urban street tree",
]

DEFAULT_NEGATIVE_PROMPTS = [
    "a photo with no tree",
    "building",
    "road",
    "pavement",
    "car",
    "person",
    "a photo of grass only",
]


def load_clip_model(
    model_path: str | Path | None = None,
    device: str | None = None,
):
    """
    Load CLIP from local TreeCo model storage.

    Priority:
    1. explicit model_path
    2. TREECO_MODELS_DIR via get_model_path("clip")
    3. <TreeCo repo>/IMAGE_MODELS via get_model_path("clip")
    """
    if model_path is None:
        model_path = get_model_path("clip")

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"CLIP model folder not found:\n"
            f"  {model_path}\n\n"
            f"Run:\n"
            f"  treeco-download-models"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    model = CLIPModel.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device).eval()

    print(f"Loaded CLIP model from {model_path} on {device}")

    return model, processor, device


def load_image(image_or_path: ImageLike) -> Image.Image:
    if isinstance(image_or_path, (str, Path)):
        return Image.open(image_or_path).convert("RGB")

    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")

    raise TypeError(f"Unsupported image input type: {type(image_or_path)}")


def score_images_batch(
    image_paths: Sequence[ImageLike],
    model,
    processor,
    device: str,
    batch_size: int = 8,
    threshold: float = 0.5,
    tree_prompts: Sequence[str] | None = None,
    negative_prompts: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Score images as tree/non-tree using CLIP prompt similarity.
    Accepts paths or PIL images.
    """
    if tree_prompts is None:
        tree_prompts = DEFAULT_TREE_PROMPTS

    if negative_prompts is None:
        negative_prompts = DEFAULT_NEGATIVE_PROMPTS

    all_prompts = list(tree_prompts) + list(negative_prompts)
    n_tree = len(tree_prompts)

    results: List[Dict[str, Any]] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        images = []
        valid_items = []

        for p in batch_paths:
            try:
                img = load_image(p)
                images.append(img)
                valid_items.append(p)

            except Exception as e:
                print(f"Failed to load image {p}: {e}")

                results.append(
                    {
                        "image_path": str(p),
                        "tree_score": 0.0,
                        "top_prompt": "LOAD_ERROR",
                        "is_tree": False,
                        "raw_probs": [],
                    }
                )

        if not images:
            continue

        inputs = processor(
            text=all_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()

        for item, prob in zip(valid_items, probs):
            tree_score = float(np.sum(prob[:n_tree]))
            top_idx = int(np.argmax(prob))
            top_prompt = all_prompts[top_idx]

            results.append(
                {
                    "image_path": str(item),
                    "tree_score": tree_score,
                    "top_prompt": top_prompt,
                    "is_tree": tree_score >= threshold,
                    "raw_probs": prob.tolist(),
                }
            )

    return results