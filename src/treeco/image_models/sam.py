from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


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