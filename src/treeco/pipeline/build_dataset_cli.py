#!/usr/bin/env python3

import argparse

from treeco.image_models.download import get_model_path
from treeco.pipeline.build_dataset import build_tree_dataset


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--dataset_name", required=True)

    ap.add_argument("--keep_rgb", action="store_true", default=True)
    ap.add_argument("--no_keep_rgb", action="store_false", dest="keep_rgb")

    ap.add_argument("--keep_sam", action="store_true")
    ap.add_argument("--keep_depth", action="store_true")

    ap.add_argument("--clip_model_path", default=None)
    ap.add_argument("--clip_threshold", default=0.5, type=float)

    ap.add_argument("--dino_model_path", default=None)
    ap.add_argument("--dino_threshold", default=0.2, type=float)
    ap.add_argument("--dino_text_threshold", default=0.25, type=float)
    ap.add_argument("--dino_score_min", default=0.2, type=float)

    ap.add_argument("--depth_ckpt", default=None)

    args = ap.parse_args()

    clip_model_path = args.clip_model_path or get_model_path("clip")
    dino_model_path = args.dino_model_path or get_model_path("grounding_dino")
    depth_ckpt = args.depth_ckpt or get_model_path("depth_anything")

    build_tree_dataset(
        input_path=args.input,
        out_root=args.out_root,
        dataset_name=args.dataset_name,
        keep_rgb=args.keep_rgb,
        keep_sam=args.keep_sam,
        keep_depth=args.keep_depth,
        clip_model_path=clip_model_path,
        clip_threshold=args.clip_threshold,
        dino_model_path=dino_model_path,
        dino_threshold=args.dino_threshold,
        dino_text_threshold=args.dino_text_threshold,
        dino_score_min=args.dino_score_min,
        depth_ckpt=depth_ckpt,
    )


if __name__ == "__main__":
    main()