from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from treeco.image_models.download import get_model_path


ImageLike = Union[str, Path, Image.Image]
Box = List[int]


DEFAULT_TREE_LABELS = [[
    "tree"
]]


def load_grounding_dino(
    model_path: Union[str, Path, None] = None,
    device: Optional[str] = None,
):
    """
    Load GroundingDINO processor and model from local folder.

    Priority:
    1. explicit model_path
    2. TREECO_MODELS_DIR / downloaded GroundingDINO
    3. TreeCo/IMAGE_MODELS downloaded GroundingDINO
    """
    if model_path is None:
        model_path = get_model_path("grounding_dino")

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"GroundingDINO model folder not found:\n"
            f"  {model_path}\n\n"
            f"Run:\n"
            f"  treeco-download-models"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device).eval()

    print(f"Loaded GroundingDINO model from {model_path} on {device}")
    return processor, model, device


def load_image(image_or_path: ImageLike) -> Image.Image:
    if isinstance(image_or_path, (str, Path)):
        return Image.open(image_or_path).convert("RGB")

    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")

    raise TypeError(f"Unsupported image input type: {type(image_or_path)}")


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _area_ratio(box: Box, image_size: Tuple[int, int]) -> float:
    x0, y0, x1, y1 = box
    w_img, h_img = image_size
    img_area = max(1.0, float(w_img * h_img))
    area = max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
    return area / img_area


def select_tree_box(
    detections: Dict[str, Any],
    image_size: Tuple[int, int],
    score_min: float = 0.2,
    border_margin: int = 5,
    max_aspect_ratio: float = 6.0,
    edge_penalty_base: float = 0.65,
    max_full_image_ratio: float = 0.92,
) -> tuple[Optional[Box], Optional[float], Optional[str], bool]:
    """
    Select the best tree box.

    Logic:
    - Compute combined score from detector confidence, area, edge penalty, aspect penalty.
    - Avoid boxes that cover almost the full image if a non-full-image alternative exists.
    - If the only valid detection is full-image-sized, keep it as fallback.

    Returns:
      best_box, raw_detection_score, label, used_full_image_fallback
    """
    W, H = image_size
    candidates = []

    boxes = detections.get("boxes", [])
    scores = detections.get("scores", [])
    labels = detections.get("labels", [])

    for i, (box, score) in enumerate(zip(boxes, scores)):
        s = _safe_float(score)
        if s is None or s < score_min:
            continue

        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        w, h = x1 - x0, y1 - y0

        if w <= 1 or h <= 1:
            continue

        area = w * h
        area_ratio = area / max(1.0, float(W * H))

        touch = (
            (x0 < border_margin)
            + (y0 < border_margin)
            + (x1 > W - border_margin)
            + (y1 > H - border_margin)
        )
        edge_penalty = edge_penalty_base ** touch

        aspect = max(w / h, h / w)
        aspect_penalty = 1.0 if aspect < max_aspect_ratio else 0.5

        combined = (area * s) * edge_penalty * aspect_penalty

        label = None
        if labels is not None and i < len(labels):
            label = str(labels[i])

        candidates.append(
            {
                "box": [int(x0), int(y0), int(x1), int(y1)],
                "score": s,
                "label": label,
                "area_ratio": area_ratio,
                "combined": combined,
            }
        )

    if not candidates:
        return None, None, None, False

    candidates = sorted(candidates, key=lambda x: x["combined"], reverse=True)

    non_full = [c for c in candidates if c["area_ratio"] < max_full_image_ratio]

    if non_full:
        chosen = non_full[0]
        used_full_image_fallback = False
    else:
        chosen = candidates[0]
        used_full_image_fallback = True

    return (
        chosen["box"],
        chosen["score"],
        chosen["label"],
        used_full_image_fallback,
    )


def detect_tree_box(
    image_or_path: ImageLike,
    processor,
    model,
    device: str,
    text_labels: Optional[Sequence[Sequence[str]]] = None,
    threshold: float = 0.2,
    text_threshold: float = 0.25,
    score_min: float = 0.2,
    max_full_image_ratio: float = 0.92,
) -> Dict[str, Any]:
    """
    Run GroundingDINO on one image and return detections + selected best box.
    """
    if text_labels is None:
        text_labels = DEFAULT_TREE_LABELS

    image = load_image(image_or_path)

    inputs = processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    detections = results[0]

    best_box, best_score, best_label, used_full_fallback = select_tree_box(
        detections=detections,
        image_size=image.size,
        score_min=score_min,
        max_full_image_ratio=max_full_image_ratio,
    )

    return {
        "image": image,
        "detections": detections,
        "boxes": detections.get("boxes", []),
        "scores": detections.get("scores", []),
        "labels": detections.get("labels", []),
        "best_box": best_box,
        "best_score": best_score,
        "best_label": best_label,
        "used_full_image_fallback": used_full_fallback,
    }


def expand_box(
    box: Box,
    image_size: Tuple[int, int],
    scale: float = 1.35,
    scale_x: Optional[float] = None,
    scale_y: Optional[float] = None,
) -> Box:
    x0, y0, x1, y1 = box
    W, H = image_size

    if scale_x is None:
        scale_x = scale

    if scale_y is None:
        scale_y = scale

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    w = (x1 - x0) * scale_x
    h = (y1 - y0) * scale_y

    x0 = max(0, int(cx - w / 2))
    y0 = max(0, int(cy - h / 2))
    x1 = min(W, int(cx + w / 2))
    y1 = min(H, int(cy + h / 2))

    return [x0, y0, x1, y1]


def crop_box(image_or_path: ImageLike, box: Optional[Box]) -> Optional[Image.Image]:
    if box is None:
        return None

    image = load_image(image_or_path)
    return image.crop(box)


def show_tree_detections(
    result: Dict[str, Any],
    score_min: float = 0.2,
    figsize: Tuple[int, int] = (8, 6),
    show_selected_label: bool = True,
):
    image = result["image"]
    detections = result["detections"]
    best_box = result["best_box"]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    ax.axis("off")

    for box, score in zip(detections["boxes"], detections["scores"]):
        s = float(score)

        if s < score_min:
            continue

        x0, y0, x1, y1 = box.tolist()

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x0,
            max(y0 - 6, 0),
            f"{s:.2f}",
            color="red",
            fontsize=9,
            backgroundcolor="white",
        )

    if best_box is not None:
        x0, y0, x1, y1 = best_box

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=3,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

        if show_selected_label:
            ax.text(
                x0,
                max(y0 - 12, 0),
                "SELECTED",
                color="lime",
                fontsize=11,
                backgroundcolor="black",
            )

    plt.tight_layout()
    plt.show()