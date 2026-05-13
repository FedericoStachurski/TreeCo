MODEL_REGISTRY = {
    "clip": {
        "display_name": "CLIP ViT-B/32",
        "type": "huggingface_snapshot",
        "repo_id": "openai/clip-vit-base-patch32",
        "local_dir": "clip-vit-base-patch32",
    },
    "grounding_dino": {
        "display_name": "GroundingDINO Tiny",
        "type": "huggingface_snapshot",
        "repo_id": "IDEA-Research/grounding-dino-tiny",
        "local_dir": "grounding_dino_tiny",
    },
    "sam": {
        "display_name": "SAM ViT-B",
        "type": "url_file",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "local_dir": "sam_vit_b",
        "filename": "sam_vit_b_01ec64.pth",
    },
    "depth_anything_repo": {
        "display_name": "Depth Anything V2 Repository",
        "type": "git_clone",
        "repo_url": "https://github.com/DepthAnything/Depth-Anything-V2.git",
        "local_dir": "Depth-Anything-V2",
    },
    "depth_anything": {
        "display_name": "Depth Anything V2 ViT-B Checkpoint",
        "type": "huggingface_file",
        "repo_id": "depth-anything/Depth-Anything-V2-Base",
        "filename": "depth_anything_v2_vitb.pth",
        "local_dir": "depth_anything_v2_vitb",
    },
}