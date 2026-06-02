from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from contextlib import nullcontext
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


# ---------------------------------------------------------------------
# Legacy SAM / SAM 1
# ---------------------------------------------------------------------


def load_sam(checkpoint_path: str | Path, model_type: str = "vit_b"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    return predictor, device


def infer_sam_mask(image: Image.Image, predictor: SamPredictor, threshold: float = 0.0):
    image_np = np.array(image.convert("RGB"))

    predictor.set_image(image_np)

    full_logits, scores, low_res_logits = predictor.predict(
        box=None,
        multimask_output=True,
        return_logits=True,
    )

    best_idx = int(np.argmax(scores))

    sam_logits_full = full_logits[best_idx].astype(np.float32)
    sam_mask_full = (sam_logits_full > threshold).astype(np.uint8)

    return {
        "logits": sam_logits_full,          # full-size, aligned to original image
        "mask": sam_mask_full,              # thresholded version if needed
        "low_res_logits": low_res_logits[best_idx].astype(np.float32),
        "score": float(scores[best_idx]),
    }


# ---------------------------------------------------------------------
# SAM 3 helpers
# ---------------------------------------------------------------------


def _maybe_add_sam3_repo_to_path(sam3_repo_path: str | Path | None = None) -> None:
    """
    Make the cloned facebookresearch/sam3 repo importable.

    This lets TreeCo work either if SAM 3 was installed with:

        pip install -e /path/to/IMAGE_MODELS/sam3

    or if the repo was only cloned into IMAGE_MODELS/sam3.
    """
    if sam3_repo_path is None:
        try:
            from treeco.image_models.download import get_model_path

            sam3_repo_path = get_model_path("sam3_repo")
        except Exception:
            sam3_repo_path = None

    if sam3_repo_path is None:
        return

    sam3_repo_path = Path(sam3_repo_path).expanduser().resolve()

    if sam3_repo_path.exists() and str(sam3_repo_path) not in sys.path:
        sys.path.insert(0, str(sam3_repo_path))


def _resolve_sam3_checkpoint_path(model_path: str | Path | None) -> str | None:
    """
    Resolve a SAM 3 checkpoint file.

    The registry path for 'sam3' is usually a directory downloaded from
    Hugging Face. This function finds the actual checkpoint file inside it.
    """
    if model_path is None:
        return None

    path = Path(model_path).expanduser().resolve()

    if path.is_file():
        return str(path)

    if not path.exists():
        raise FileNotFoundError(f"SAM 3 model path does not exist:\n{path}")

    preferred_names = [
        "sam3.1_multiplex.pt",
        "sam3.1.pt",
        "sam3.pt",
        "model.pt",
        "checkpoint.pt",
    ]

    for name in preferred_names:
        candidate = path / name
        if candidate.exists():
            return str(candidate)

    candidates = []
    for pattern in ["*.pt", "*.pth", "*.safetensors"]:
        candidates.extend(path.rglob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "Could not find a SAM 3 checkpoint file inside:\n"
            f"{path}\n\n"
            "Expected one of: .pt, .pth, or .safetensors"
        )

    # Prefer the largest file, because checkpoints are usually much larger
    # than small metadata/test files.
    candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
    return str(candidates[0])


def _to_numpy(x):
    """
    Convert SAM 3 output tensors/lists to numpy arrays.

    Important:
    PyTorch bfloat16 tensors cannot be converted directly to NumPy.
    Cast bfloat16/float16 tensors to float32 first.
    """
    if x is None:
        return np.array([])

    if isinstance(x, np.ndarray):
        return x

    if torch.is_tensor(x):
        x = x.detach()

        if x.dtype in {torch.bfloat16, torch.float16}:
            x = x.float()

        return x.cpu().numpy()

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.array([])

        if torch.is_tensor(x[0]):
            stacked = torch.stack(list(x)).detach()

            if stacked.dtype in {torch.bfloat16, torch.float16}:
                stacked = stacked.float()

            return stacked.cpu().numpy()

        return np.asarray(x)

    return np.asarray(x)


def _normalise_sam3_masks(masks: Any) -> np.ndarray:
    """
    Normalise SAM 3 masks to shape:

        N x H x W

    Handles common output shapes:
        H x W
        N x H x W
        N x 1 x H x W
    """
    masks_np = _to_numpy(masks)

    if masks_np.size == 0:
        return np.empty((0, 0, 0), dtype=np.uint8)

    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]

    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, :, :]

    if masks_np.ndim != 3:
        raise ValueError(f"Unexpected SAM 3 mask shape: {masks_np.shape}")

    return masks_np.astype(bool).astype(np.uint8)


def _normalise_sam3_boxes(boxes: Any) -> np.ndarray:
    """
    Normalise SAM 3 boxes to shape:

        N x 4
    """
    boxes_np = _to_numpy(boxes)

    if boxes_np.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    boxes_np = boxes_np.astype(np.float32)

    if boxes_np.ndim == 1 and boxes_np.shape[0] == 4:
        boxes_np = boxes_np[None, :]

    if boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
        raise ValueError(f"Unexpected SAM 3 box shape: {boxes_np.shape}")

    return boxes_np


def _normalise_sam3_scores(scores: Any, n: int) -> np.ndarray:
    """
    Normalise SAM 3 scores to shape:

        N
    """
    scores_np = _to_numpy(scores)

    if scores_np.size == 0:
        return np.full((n,), np.nan, dtype=np.float32)

    scores_np = scores_np.astype(np.float32).reshape(-1)

    if len(scores_np) < n:
        padded = np.full((n,), np.nan, dtype=np.float32)
        padded[: len(scores_np)] = scores_np
        return padded

    return scores_np[:n]


def _box_from_mask(mask: np.ndarray) -> list[float] | None:
    """
    Build xyxy box from a binary mask.
    """
    ys, xs = np.where(mask.astype(bool))

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max() + 1)
    y2 = float(ys.max() + 1)

    return [x1, y1, x2, y2]


def load_sam3(
    model_path: str | Path | None = None,
    sam3_repo_path: str | Path | None = None,
    device: str | None = None,
    compile_model: bool = False,
):
    """
    Load SAM 3 image model and processor.

    Parameters
    ----------
    model_path:
        Path to the SAM 3 checkpoint file or downloaded Hugging Face snapshot
        directory, e.g. IMAGE_MODELS/sam3_1.

    sam3_repo_path:
        Path to the cloned facebookresearch/sam3 repo, e.g. IMAGE_MODELS/sam3.
        Optional if the package was installed with pip install -e.

    device:
        "cuda" or "cpu". Defaults to CUDA if available.

    Returns
    -------
    model, processor, device
    """
    _maybe_add_sam3_repo_to_path(sam3_repo_path)

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:
        raise ImportError(
            "Could not import SAM 3.\n\n"
            "Make sure the SAM 3 repo has been cloned and installed, e.g.:\n"
            "  treeco-download-models\n"
            "  pip install -e /home/fss6k/TreeCo/IMAGE_MODELS/sam3\n\n"
            "Original import error:\n"
            f"{e}"
        ) from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_sam3_checkpoint_path(model_path)

    build_kwargs = {
        "device": device,
        "eval_mode": True,
        "compile": compile_model,
    }

    if checkpoint_path is not None:
        build_kwargs["checkpoint_path"] = checkpoint_path
        build_kwargs["load_from_HF"] = False
    else:
        build_kwargs["load_from_HF"] = True

    model = build_sam3_image_model(**build_kwargs)
    model.eval()

    processor = Sam3Processor(model)

    return model, processor, device


@torch.inference_mode()
def infer_sam3_tree_candidates(
    image: Image.Image,
    model,
    processor,
    device: str | None = None,
    prompt: str = "tree",
    min_score: float | None = None,
) -> dict:
    """
    Run SAM 3 concept segmentation on one image using a text prompt.

    Returns a TreeCo-friendly structure:

        {
            "prompt": "tree",
            "candidates": [
                {
                    "box": [x1, y1, x2, y2],
                    "score": 0.91,
                    "mask": np.ndarray[H, W],
                    "area_pixels": 12345,
                    "label": "tree",
                },
                ...
            ],
            "raw_output": output,
        }

    In the dataset builder you can select the candidate with the largest
    area and use its box as the crop source.
    """
    image = image.convert("RGB")

    use_cuda_amp = (
    device is not None
    and str(device).startswith("cuda")
    and torch.cuda.is_available()
    )

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_cuda_amp
        else nullcontext()
    )

    with amp_context:
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )
        
    masks_np = _normalise_sam3_masks(output.get("masks"))
    boxes_np = _normalise_sam3_boxes(output.get("boxes"))

    n = max(len(masks_np), len(boxes_np))

    if n == 0:
        return {
            "prompt": prompt,
            "candidates": [],
            "raw_output": output,
        }

    scores_np = _normalise_sam3_scores(output.get("scores"), n=n)

    candidates = []

    for i in range(n):
        mask = masks_np[i] if i < len(masks_np) else None

        if i < len(boxes_np):
            box = boxes_np[i].astype(float).tolist()
        elif mask is not None:
            box = _box_from_mask(mask)
        else:
            box = None

        score = scores_np[i]
        score_float = None if np.isnan(score) else float(score)

        if min_score is not None and score_float is not None and score_float < min_score:
            continue

        if mask is not None:
            area_pixels = float(mask.astype(bool).sum())
        elif box is not None:
            x1, y1, x2, y2 = box
            area_pixels = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        else:
            area_pixels = 0.0

        candidates.append(
            {
                "box": box,
                "score": score_float,
                "mask": mask,
                "area_pixels": area_pixels,
                "label": prompt,
            }
        )

    return {
        "prompt": prompt,
        "candidates": candidates,
        "raw_output": output,
    }