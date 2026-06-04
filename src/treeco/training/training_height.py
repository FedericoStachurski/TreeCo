#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path


import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score, accuracy_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


INPUT_CHANNELS = {
    "rgb": 3,
    "rgb_depth": 4,
    "rgb_sam": 4,
    "rgb_sam_depth": 5,
    "rgb_sam3": 4,
    "rgb_sam3_depth": 5,

}


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()

class TreeHeightDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        input_mode: str = "rgb",
        image_source: str = "crop",
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.input_mode = input_mode
        self.image_source = image_source
        self.train = train

        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.use_depth = input_mode in {
            "rgb_depth",
            "rgb_sam_depth",
            "rgb_sam3_depth",
            "sam3_depth",
        }

        self.use_sam = input_mode in {
            "rgb_sam",
            "rgb_sam_depth",
        }

        self.use_sam3 = input_mode in {
            "rgb_sam3",
            "rgb_sam3_depth",
        }

        if image_source == "crop":
            self.rgb_col = "RGB_CROP_PATH"
        elif image_source == "full":
            self.rgb_col = "ORIGINAL_RGB_PATH"
        else:
            raise ValueError(f"Unknown image_source: {image_source}")

        self.color_jitter = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

        self.rgb_erasing = transforms.RandomErasing(
            p=0.20,
            scale=(0.02, 0.10),
            ratio=(0.3, 3.3),
            value="random",
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_single_channel_image(self, path: str) -> Image.Image:
        try:
            arr = np.load(path).astype(np.float32)

            if arr.ndim == 3:
                arr = np.squeeze(arr)

            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")

            amin = np.nanmin(arr)
            amax = np.nanmax(arr)

            arr = (arr - amin) / (amax - amin + 1e-8)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

            img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")

        except Exception:
            img = Image.fromarray(
                np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            ).convert("L")

        return img

    def _apply_shared_geometric_transforms(
        self,
        rgb: Image.Image,
        single_channels: list[Image.Image],
    ):
        # Resize all modalities first
        rgb = TF.resize(rgb, [self.image_size, self.image_size])
        single_channels = [
            TF.resize(ch, [self.image_size, self.image_size])
            for ch in single_channels
        ]

        if self.train:
            # Same horizontal flip for RGB, SAM, depth
            if random.random() < 0.5:
                rgb = TF.hflip(rgb)
                single_channels = [TF.hflip(ch) for ch in single_channels]

            # Same rotation angle for RGB, SAM, depth
            angle = random.uniform(-8, 8)
            rgb = TF.rotate(
                rgb,
                angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )

            single_channels = [
                TF.rotate(
                    ch,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )
                for ch in single_channels
            ]

        return rgb, single_channels

    def _rgb_to_tensor(self, rgb: Image.Image) -> torch.Tensor:
        if self.train:
            rgb = self.color_jitter(rgb)

        rgb_tensor = TF.to_tensor(rgb)
        rgb_tensor = TF.normalize(
            rgb_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.train:
            rgb_tensor = self.rgb_erasing(rgb_tensor)

        return rgb_tensor

    def _single_to_tensor(
        self,
        img: Image.Image,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = TF.to_tensor(img).to(dtype=dtype)
        return tensor

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rgb = Image.open(row[self.rgb_col]).convert("RGB")

        single_channels = []

        if self.use_sam:
            single_channels.append(
                self._load_single_channel_image(row["SAM_LOGITS_PATH"])
            )
        
        if self.use_sam3:
            single_channels.append(
                self._load_single_channel_image(row["SAM3_MASK_PATH"])
            )

        if self.use_depth:
            single_channels.append(
                self._load_single_channel_image(row["DEPTH_PATH"])
            )

        rgb, single_channels = self._apply_shared_geometric_transforms(
            rgb,
            single_channels,
        )

        rgb_tensor = self._rgb_to_tensor(rgb)

        channels = [rgb_tensor]

        for ch in single_channels:
            channels.append(
                self._single_to_tensor(
                    ch,
                    dtype=rgb_tensor.dtype,
                )
            )

        x = torch.cat(channels, dim=0)
        y = int(row["HEIGHT_CLASS_IDX"])

        return x, y


def build_resnet(
    backbone: str,
    num_classes: int,
    in_channels: int,
    device: torch.device,
    dropout_rate: float = 0.1,
) -> nn.Module:
    backbone = backbone.lower()

    weights_map = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
    }

    if backbone not in weights_map:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model = getattr(models, backbone)(weights=weights_map[backbone])

    if in_channels != 3:
        old_conv = model.conv1

        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight

            for c in range(3, in_channels):
                model.conv1.weight[:, c:c + 1] = old_conv.weight.mean(
                    dim=1,
                    keepdim=True,
                )

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, num_classes),
    )

    return model.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    optimizer=None,
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)

            if train_mode:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, all_preds, all_labels


def find_dataset_dir(
    out_root: Path,
    dataset_name: str | None,
    dataset_path: str | None,
) -> Path:
    if dataset_path is not None:
        p = Path(dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset path not found: {p}")
        return p

    if dataset_name is None:
        raise ValueError("Provide either --dataset_path or --dataset_name")

    matches = sorted([p for p in out_root.glob(f"{dataset_name}_*") if p.is_dir()])

    if not matches:
        raise FileNotFoundError(
            f"No dataset folders found for pattern {dataset_name}_* under {out_root}"
        )

    return matches[-1]


def load_manifest(dataset_dir: Path) -> pd.DataFrame:
    manifest_path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = [
        "RGB_CROP_PATH",
        "HEIGHT_CLASS_STR",
        "HEIGHT_CLASS_IDX",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    optional_cols = [
        "ORIGINAL_RGB_PATH",
        "SAM_LOGITS_PATH",
        "SAM3_MASK_PATH",
        "DEPTH_PATH",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "TRAINABLE" in df.columns:
        df = df[df["TRAINABLE"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    df = df[
        df["HEIGHT_CLASS_STR"].notna()
        & df["HEIGHT_CLASS_IDX"].notna()
    ].copy()

    df["HEIGHT_CLASS_IDX_ORIGINAL"] = df["HEIGHT_CLASS_IDX"].astype(int)

    class_order = (
        df[["HEIGHT_CLASS_IDX_ORIGINAL", "HEIGHT_CLASS_STR"]]
        .drop_duplicates()
        .sort_values("HEIGHT_CLASS_IDX_ORIGINAL")
    )

    remap = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(class_order["HEIGHT_CLASS_IDX_ORIGINAL"])
    }

    df["HEIGHT_CLASS_IDX"] = df["HEIGHT_CLASS_IDX_ORIGINAL"].map(remap).astype(int)

    print("\nHeight class remapping:")
    for _, row in class_order.iterrows():
        old_idx = int(row["HEIGHT_CLASS_IDX_ORIGINAL"])
        new_idx = remap[old_idx]
        label = row["HEIGHT_CLASS_STR"]
        print(f"  original {old_idx} -> train {new_idx}: {label}")

    return df.reset_index(drop=True)

def filter_manifest_for_inputs(
    df: pd.DataFrame,
    input_mode: str,
    image_source: str,
) -> pd.DataFrame:
    df = df.copy()

    if image_source == "crop":
        rgb_col = "RGB_CROP_PATH"
    elif image_source == "full":
        rgb_col = "ORIGINAL_RGB_PATH"
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    # RGB existence
    df = df[df[rgb_col].notna()].copy()
    df = df[df[rgb_col].apply(lambda p: Path(str(p)).exists())].copy()

    # Depth requirements
    if input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}:
        df = df[df["DEPTH_PATH"].notna()].copy()
        df = df[
            df["DEPTH_PATH"].apply(lambda p: Path(str(p)).exists())
        ].copy()

    # SAM requirements
    if input_mode in {"rgb_sam", "rgb_sam_depth"}:
        df = df[df["SAM_LOGITS_PATH"].notna()].copy()
        df = df[
            df["SAM_LOGITS_PATH"].apply(lambda p: Path(str(p)).exists())
        ].copy()

    #SAM3 requirements
    if input_mode in {"rgb_sam3", "rgb_sam3_depth"}:
        df = df[df["SAM3_MASK_PATH"].notna()].copy()
        df = df[
            df["SAM3_MASK_PATH"].apply(lambda p: Path(str(p)).exists())
        ].copy()    

    df = df.drop_duplicates(subset=[rgb_col]).reset_index(drop=True)

    return df


def make_class_mapping(df: pd.DataFrame) -> dict[str, str]:
    mapping_df = (
        df[["HEIGHT_CLASS_IDX", "HEIGHT_CLASS_STR"]]
        .drop_duplicates()
        .sort_values("HEIGHT_CLASS_IDX")
    )

    return {
        str(int(row["HEIGHT_CLASS_IDX"])): str(row["HEIGHT_CLASS_STR"])
        for _, row in mapping_df.iterrows()
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--dataset_path", type=str, default=None)

    ap.add_argument(
        "--out_dir",
        type=str,
        default="TreeCo/models",
    )

    ap.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional custom run name.",
    )

    ap.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )

    ap.add_argument(
        "--input_mode",
        type=str,
        default="rgb",
        choices=["rgb", "rgb_depth", "rgb_sam", "rgb_sam_depth", "rgb_sam3", "rgb_sam3_depth", "sam3_depth"],
        help="Model inputs: RGB only, RGB+Depth, RGB+SAM, or RGB+SAM+Depth.",
    )

    ap.add_argument(
        "--image_source",
        type=str,
        default="full",
        choices=["crop", "full"],
        help="Use DINO-cropped RGB image or full original RGB image.",
    )

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dropout_rate", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--criterion", type=str, default="cross_entropy", choices=["cross_entropy", "focal", "weighted_ce"])
    ap.add_argument("--scheduler", type=str, default=None, choices=["none", "plateau", "cosine", "step", "onecycle"])

    args = ap.parse_args()

    seed_everything(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)

    dataset_dir = find_dataset_dir(
        out_root=Path("."),
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )

    df = load_manifest(dataset_dir)

    df = filter_manifest_for_inputs(
        df,
        input_mode=args.input_mode,
        image_source=args.image_source,
    )

    if df.empty:
        raise RuntimeError(
            f"No valid training rows after filtering for "
            f"input_mode={args.input_mode}, image_source={args.image_source}."
        )

    print(f"Using dataset: {dataset_dir}")
    print(f"Device: {device}")
    print(f"Input mode: {args.input_mode}")
    print(f"Image source: {args.image_source}")
    print(f"Total rows: {len(df)}")
    print("\nFull class counts:")
    print(df["HEIGHT_CLASS_STR"].value_counts().sort_index())

# ---------------------------------------------------
# Tree-level split (prevents leakage across images)
# ---------------------------------------------------

    tree_df = (
        df[["ID", "HEIGHT_CLASS_IDX"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    train_tree_ids, val_tree_ids = train_test_split(
        tree_df["ID"],
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=tree_df["HEIGHT_CLASS_IDX"],
    )

    train_df = df[df["ID"].isin(train_tree_ids)].copy()
    val_df = df[df["ID"].isin(val_tree_ids)].copy()

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("\nUnique trees:")
    print(f"Train trees: {train_df['ID'].nunique()}")
    print("\nTrain class counts:")
    print(train_df["HEIGHT_CLASS_STR"].value_counts().sort_index())
    
    print("\nValidation class counts:")
    print(val_df["HEIGHT_CLASS_STR"].value_counts().sort_index())
    print(f"Val trees: {val_df['ID'].nunique()}")

    print("\nImages:")
    print(f"Train images: {len(train_df)}")
    print(f"Val images: {len(val_df)}")

    classes = np.sort(train_df["HEIGHT_CLASS_IDX"].unique())

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["HEIGHT_CLASS_IDX"],
    )

    class_weights = torch.tensor(
        class_weights_np,
        dtype=torch.float32,
        device=device,
    )

    train_ds = TreeHeightDataset(
        train_df,
        image_size=args.image_size,
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=True,
    )

    val_ds = TreeHeightDataset(
        val_df,
        image_size=args.image_size,
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    in_channels = INPUT_CHANNELS[args.input_mode]
    num_classes = int(df["HEIGHT_CLASS_IDX"].nunique())

    model = build_resnet(
        backbone=args.backbone,
        num_classes=num_classes,
        in_channels=in_channels,
        device=device,
        dropout_rate=args.dropout_rate,
    )

    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1,
        )
    elif args.criterion == "focal":
        criterion = FocalLoss(
            weight=class_weights,
            alpha=1,
            gamma=2,
        )
    elif args.criterion == "weighted_ce":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5,
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * len(train_loader),
        )
    else:
        scheduler = None

    models_root = out_root / "tree_height_models"
    models_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_name is not None:
        run_name = f"{args.run_name}_{timestamp}"
    else:
        run_name = (
            f"height_{args.backbone}_"
            f"{args.input_mode}_{args.image_source}_{timestamp}"
        )

    run_dir = models_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pth"
    last_model_path = run_dir / "last_model.pth"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"

    class_mapping = make_class_mapping(df)

    config = {
        "task": "height_classification",
        "dataset_dir": str(dataset_dir),
        "backbone": args.backbone,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "input_mode": args.input_mode,
        "image_source": args.image_source,
        "use_depth": args.input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth", "sam3_depth"},
        "use_sam": args.input_mode in {"rgb_sam", "rgb_sam_depth", "rgb_sam3", "sam3_depth"},
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "num_workers": args.num_workers,
        "class_weighted_loss": True,
        "device": str(device),
        "height_class_mapping": class_mapping,
        "scheduler": args.scheduler if hasattr(args, "scheduler") else "none",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_recall_macro": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1

    print(f"\nSaving training run to: {run_dir}")

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_loss, val_acc, val_preds, val_labels = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        val_f1_macro = f1_score(
            val_labels,
            val_preds,
            average="macro",
            zero_division=0,
        )

        val_recall_macro = recall_score(
            val_labels,
            val_preds,
            average="macro",
            zero_division=0,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1_macro)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_f1_macro"].append(float(val_f1_macro))
        history["val_recall_macro"].append(float(val_recall_macro))
        history["lr"].append(float(current_lr))

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"val_f1={val_f1_macro:.4f} val_recall={val_recall_macro:.4f} | "
            f"lr={current_lr:.4e}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "backbone": args.backbone,
            "num_classes": num_classes,
            "in_channels": in_channels,
            "input_mode": args.input_mode,
            "image_source": args.image_source,
            "image_size": args.image_size,
            "height_class_mapping": class_mapping,
            "val_accuracy": float(val_acc),
            "val_f1_macro": float(val_f1_macro),
            "val_recall_macro": float(val_recall_macro),
            "scheduler": args.scheduler if hasattr(args, "scheduler") else "none",
        }

        torch.save(checkpoint, last_model_path)

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_epoch = epoch + 1
            torch.save(checkpoint, best_model_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_val_f1),
        "final_val_accuracy": float(history["val_acc"][-1]),
        "final_val_f1_macro": float(history["val_f1_macro"][-1]),
        "final_val_recall_macro": float(history["val_recall_macro"][-1]),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    np.save(run_dir / "val_labels.npy", val_labels)
    np.save(run_dir / "val_preds.npy", val_preds)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro F1: {best_val_f1:.4f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved last model to: {last_model_path}")


if __name__ == "__main__":
    main()