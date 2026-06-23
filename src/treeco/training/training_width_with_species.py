#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


SPECIES_UNKNOWN = "Unknown"
SPECIES_RARE = "Rare/Other"


def _is_missing_text(value) -> bool:
    if value is None:
        return True

    s = str(value).strip()

    if s == "":
        return True

    return s.lower() in {"nan", "none", "null", "na", "n/a"}


import re

SPECIES_UNKNOWN = "Unknown"


def clean_species_label(row: pd.Series) -> str:
    """
    Creates one cleaned species label from TREE_TYPE and OTHER_TREE.

    Rule
    ----
    - If TREE_TYPE is "Other", use OTHER_TREE.
    - Otherwise use TREE_TYPE.
    - Missing / very unclear values become "Unknown".
    - Scientific names in brackets are removed for cleaner grouping.
    """

    tree_type = row.get("TREE_TYPE", np.nan)
    other_tree = row.get("OTHER_TREE", np.nan)

    if _is_missing_text(tree_type):
        raw = other_tree
    elif str(tree_type).strip().lower() == "other":
        raw = other_tree
    else:
        raw = tree_type

    if _is_missing_text(raw):
        return SPECIES_UNKNOWN

    label = str(raw).strip()

    # Remove scientific names or extra notes in brackets:
    # "Swedish Whitebeam (Sorbus intermedia)" -> "Swedish Whitebeam"
    label = re.sub(r"\s*\([^)]*\)", "", label)

    # Replace commas with spaces, remove repeated spaces
    label = label.replace(",", " ")
    label = re.sub(r"\s+", " ", label).strip()

    # Remove trailing punctuation
    label = label.strip(" .;:-_/")

    # Very short labels are usually typos / unusable free text.
    if len(label) < 3:
        return SPECIES_UNKNOWN

    # Standardise capitalisation:
    # "Highclere holly" -> "Highclere Holly"
    # "sycamore" -> "Sycamore"
    label = label.title()

    # Fix common botanical lowercase conventions after title-casing
    label = label.replace(" X ", " x ")

    # Manual corrections for obvious messy labels
    manual_map = {
        "Swe Wh": SPECIES_UNKNOWN,
        "N/A": SPECIES_UNKNOWN,
        "Na": SPECIES_UNKNOWN,
        "Unknown": SPECIES_UNKNOWN,

        # Optional canonical aliases
        "Highclere Holly": "Highclere Holly",
        "Swedish Whitebeam": "Swedish Whitebeam",
        "Purple Crabapple": "Purple Crabapple",
    }

    label = manual_map.get(label, label)

    if _is_missing_text(label):
        return SPECIES_UNKNOWN

    return label


def prepare_species_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    min_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Builds a species vocabulary from the training split only, then applies it
    to train and validation.

    Rare training species and unseen validation species are mapped to Rare/Other.
    This avoids leaking validation-only categories into the training vocabulary.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()

    if "TREE_SPECIES_RAW" not in train_df.columns:
        train_df["TREE_SPECIES_RAW"] = train_df.apply(clean_species_label, axis=1)

    if "TREE_SPECIES_RAW" not in val_df.columns:
        val_df["TREE_SPECIES_RAW"] = val_df.apply(clean_species_label, axis=1)

    counts = train_df["TREE_SPECIES_RAW"].value_counts(dropna=False)

    keep_species = {
        species
        for species, count in counts.items()
        if species != SPECIES_UNKNOWN and count >= min_count
    }

    def map_label(raw) -> str:
        if _is_missing_text(raw):
            return SPECIES_UNKNOWN

        raw = str(raw).strip()

        if raw == SPECIES_UNKNOWN:
            return SPECIES_UNKNOWN

        if raw in keep_species:
            return raw

        return SPECIES_RARE

    train_df["TREE_SPECIES_LABEL"] = train_df["TREE_SPECIES_RAW"].apply(map_label)
    val_df["TREE_SPECIES_LABEL"] = val_df["TREE_SPECIES_RAW"].apply(map_label)

    classes = [SPECIES_UNKNOWN, SPECIES_RARE]
    classes += sorted(
        [
            species
            for species in train_df["TREE_SPECIES_LABEL"].unique()
            if species not in {SPECIES_UNKNOWN, SPECIES_RARE}
        ]
    )

    species_to_idx = {species: idx for idx, species in enumerate(classes)}

    train_df["TREE_SPECIES_IDX"] = train_df["TREE_SPECIES_LABEL"].map(species_to_idx)
    val_df["TREE_SPECIES_IDX"] = val_df["TREE_SPECIES_LABEL"].map(species_to_idx)

    train_df["TREE_SPECIES_IDX"] = train_df["TREE_SPECIES_IDX"].fillna(
        species_to_idx[SPECIES_RARE]
    ).astype(int)

    val_df["TREE_SPECIES_IDX"] = val_df["TREE_SPECIES_IDX"].fillna(
        species_to_idx[SPECIES_RARE]
    ).astype(int)

    return train_df, val_df, species_to_idx


class TreeDBHDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        input_mode: str = "rgb",
        image_source: str = "crop",
        train: bool = True,
        use_log1p: bool = False,
        use_species: bool = False,
        species_dropout: float = 0.0,
        species_unknown_idx: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.input_mode = input_mode
        self.image_source = image_source
        self.use_log1p = use_log1p
        self.train = train
        self.use_species = use_species
        self.species_dropout = species_dropout
        self.species_unknown_idx = species_unknown_idx

        if self.use_species and "TREE_SPECIES_IDX" not in self.df.columns:
            raise ValueError(
                "use_species=True, but TREE_SPECIES_IDX is missing from the dataframe."
            )

        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.use_depth = input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}
        self.use_sam = input_mode in {"rgb_sam", "rgb_sam_depth"}
        self.use_sam3 = input_mode in {"rgb_sam3", "rgb_sam3_depth"}

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
        rgb = TF.resize(rgb, [self.image_size, self.image_size])
        single_channels = [
            TF.resize(ch, [self.image_size, self.image_size])
            for ch in single_channels
        ]

        if self.train:
            if random.random() < 0.5:
                rgb = TF.hflip(rgb)
                single_channels = [TF.hflip(ch) for ch in single_channels]

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

    def _single_to_tensor(self, img: Image.Image, dtype: torch.dtype) -> torch.Tensor:
        return TF.to_tensor(img).to(dtype=dtype)

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
            channels.append(self._single_to_tensor(ch, dtype=rgb_tensor.dtype))

        x = torch.cat(channels, dim=0)

        dbh_cm = float(row["DBH_CM"])

        if self.use_log1p:
            target = np.log1p(dbh_cm)
        else:
            target = dbh_cm

        y = torch.tensor(target, dtype=torch.float32)

        if self.use_species:
            species_idx = int(row["TREE_SPECIES_IDX"])

            # During training, occasionally mask species to Unknown so the model
            # does not become too dependent on the metadata.
            if self.train and self.species_dropout > 0:
                if random.random() < self.species_dropout:
                    species_idx = self.species_unknown_idx

            species_idx = torch.tensor(species_idx, dtype=torch.long)

            return x, species_idx, y

        return x, y


class ResNetDBHWithSpecies(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        image_feat_dim: int,
        num_species: int,
        species_emb_dim: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.image_encoder = image_encoder

        self.species_embedding = nn.Embedding(
            num_embeddings=num_species,
            embedding_dim=species_emb_dim,
        )

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(image_feat_dim + species_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(x)
        species_feat = self.species_embedding(species_idx)

        fused = torch.cat([image_feat, species_feat], dim=1)

        return self.head(fused)


def build_resnet_encoder(
    backbone: str,
    in_channels: int,
    device: torch.device,
) -> tuple[nn.Module, int]:
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

    image_feat_dim = model.fc.in_features
    model.fc = nn.Identity()

    return model.to(device), image_feat_dim


def build_resnet(
    backbone: str,
    in_channels: int,
    device: torch.device,
    dropout_rate: float = 0.1,
) -> nn.Module:
    image_encoder, image_feat_dim = build_resnet_encoder(
        backbone=backbone,
        in_channels=in_channels,
        device=device,
    )

    image_encoder.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(image_feat_dim, 1),
    )

    return image_encoder.to(device)


def build_resnet_with_species(
    backbone: str,
    in_channels: int,
    device: torch.device,
    num_species: int,
    species_emb_dim: int = 16,
    dropout_rate: float = 0.1,
) -> nn.Module:
    image_encoder, image_feat_dim = build_resnet_encoder(
        backbone=backbone,
        in_channels=in_channels,
        device=device,
    )

    model = ResNetDBHWithSpecies(
        image_encoder=image_encoder,
        image_feat_dim=image_feat_dim,
        num_species=num_species,
        species_emb_dim=species_emb_dim,
        dropout_rate=dropout_rate,
    )

    return model.to(device)

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    optimizer=None,
    use_log1p: bool = False,
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_preds_raw = []
    all_targets_raw = []

    for batch in loader:
        if len(batch) == 3:
            x, species_idx, y = batch
            species_idx = species_idx.to(device, non_blocking=True)
        else:
            x, y = batch
            species_idx = None

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            if species_idx is None:
                preds = model(x).squeeze(1)
            else:
                preds = model(x, species_idx).squeeze(1)

            # Loss is computed in the target space:
            # raw DBH if use_log1p=False, log1p(DBH) if use_log1p=True.
            loss = criterion(preds, y)

            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_preds_raw.extend(preds.detach().cpu().numpy())
        all_targets_raw.extend(y.detach().cpu().numpy())

    all_preds_raw = np.array(all_preds_raw, dtype=np.float32)
    all_targets_raw = np.array(all_targets_raw, dtype=np.float32)

    avg_loss = total_loss / max(len(all_targets_raw), 1)

    # Metrics are always computed in real DBH cm.
    if use_log1p:
        all_preds = np.expm1(all_preds_raw)
        all_targets = np.expm1(all_targets_raw)

        # Prevent negative DBH values if the network predicts log values below 0.
        all_preds = np.clip(all_preds, 0, None)
        all_targets = np.clip(all_targets, 0, None)
    else:
        all_preds = all_preds_raw
        all_targets = all_targets_raw

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = mean_squared_error(all_targets, all_preds) ** 0.5

    try:
        r2 = r2_score(all_targets, all_preds)
    except Exception:
        r2 = np.nan

    return avg_loss, mae, rmse, r2, all_preds, all_targets


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
        "ID",
        "RGB_CROP_PATH",
        "DBH_CM",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    optional_cols = [
        "ORIGINAL_RGB_PATH",
        "SAM_LOGITS_PATH",
        "DEPTH_PATH",
        "SAM3_MASK_PATH",
        "TREE_TYPE",
        "OTHER_TREE",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "TRAINABLE" in df.columns:
        df = df[
            df["TRAINABLE"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        ].copy()

    df["DBH_CM"] = pd.to_numeric(df["DBH_CM"], errors="coerce")
    df = df[df["DBH_CM"].notna()].copy()

    # Optional sanity filter: removes zero/negative weird labels
    df = df[df["DBH_CM"] > 0].copy()

    

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

    df = df[df[rgb_col].notna()].copy()
    df = df[df[rgb_col].apply(lambda p: Path(str(p)).exists())].copy()

    if input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}:
        df = df[df["DEPTH_PATH"].notna()].copy()
        df = df[df["DEPTH_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    if input_mode in {"rgb_sam", "rgb_sam_depth"}:
        df = df[df["SAM_LOGITS_PATH"].notna()].copy()
        df = df[df["SAM_LOGITS_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    if input_mode in {"rgb_sam3", "rgb_sam3_depth"}:
        df = df[df["SAM3_MASK_PATH"].notna()].copy()
        df = df[df["SAM3_MASK_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    df = df.drop_duplicates(subset=[rgb_col]).reset_index(drop=True)

    return df


def make_regression_stratification_bins(tree_df: pd.DataFrame) -> pd.Series | None:
    """
    Creates temporary DBH bins for tree-level stratified splitting.
    Falls back to None if there are too few examples per bin.
    """
    try:
        bins = pd.qcut(
            tree_df["DBH_CM"],
            q=min(5, tree_df["DBH_CM"].nunique()),
            duplicates="drop",
        )

        counts = bins.value_counts()

        if len(counts) < 2 or counts.min() < 2:
            return None

        return bins.astype(str)

    except Exception:
        return None


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
        choices=["rgb", "rgb_depth", "rgb_sam", "rgb_sam_depth", "rgb_sam3", "rgb_sam3_depth"],
    )

    ap.add_argument(
        "--image_source",
        type=str,
        default="full",
        choices=["crop", "full"],
    )

    ap.add_argument("--use_log1p", action="store_true",
                    help="Whether to apply log1p transformation to the target DBH values. " \
                    "This can help stabilize training when there is a wide range of DBH values.")

    ap.add_argument(
        "--use_species",
        action="store_true",
        help="Add TREE_TYPE / OTHER_TREE as a categorical metadata branch.",
    )
    ap.add_argument(
        "--species_emb_dim",
        type=int,
        default=16,
        help="Embedding dimension for the species metadata branch.",
    )
    ap.add_argument(
        "--species_min_count",
        type=int,
        default=5,
        help="Minimum training examples required to keep a species as its own category.",
    )
    ap.add_argument(
        "--species_dropout",
        type=float,
        default=0.10,
        help="Probability of replacing species with Unknown during training.",
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
    ap.add_argument("--criterion", type=str, default="huber", choices=["huber", "smoothl1"])
    ap.add_argument("--scheduler", type=str, default=None, choices=["none", "plateau", "cosine", "step"])

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

    # This is harmless even when --use_species is not used, and useful for diagnostics.
    df["TREE_SPECIES_RAW"] = df.apply(clean_species_label, axis=1)

    if df.empty:
        raise RuntimeError(
            f"No valid training rows after filtering for "
            f"input_mode={args.input_mode}, image_source={args.image_source}."
        )

    print(f"Using dataset: {dataset_dir}")
    print(f"Device: {device}")
    print(f"Input mode: {args.input_mode}")
    print(f"Image source: {args.image_source}")
    print(f"Total labelled image rows: {len(df)}")
    print(f"Unique trees: {df['ID'].nunique()}")

    print("\nDBH_CM summary:")
    print(df["DBH_CM"].describe())

    # ---------------------------------------------------
    # Tree-level split to prevent leakage across images
    # ---------------------------------------------------

    tree_df = (
        df.groupby("ID", as_index=False)
        .agg(DBH_CM=("DBH_CM", "mean"))
        .reset_index(drop=True)
    )

    if args.use_log1p:
        tree_df["DBH_CM"] = np.log1p(tree_df["DBH_CM"])

    stratify_bins = make_regression_stratification_bins(tree_df)

    train_tree_ids, val_tree_ids = train_test_split(
        tree_df["ID"],
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=stratify_bins,
    )

    train_df = df[df["ID"].isin(train_tree_ids)].copy().reset_index(drop=True)
    val_df = df[df["ID"].isin(val_tree_ids)].copy().reset_index(drop=True)

    species_to_idx = None

    if args.use_species:
        train_df, val_df, species_to_idx = prepare_species_encoding(
            train_df=train_df,
            val_df=val_df,
            min_count=args.species_min_count,
        )

        print("\nSpecies metadata enabled:")
        print(f"Species categories: {len(species_to_idx)}")
        print(f"Species min count: {args.species_min_count}")
        print(f"Species embedding dim: {args.species_emb_dim}")
        print(f"Species dropout: {args.species_dropout}")
        print("\nTop training species labels:")
        print(train_df["TREE_SPECIES_LABEL"].value_counts().head(20))

    print("\nSplit:")
    print(f"Train trees: {train_df['ID'].nunique()}")
    print(f"Val trees: {val_df['ID'].nunique()}")
    print(f"Train images: {len(train_df)}")
    print(f"Val images: {len(val_df)}")

    print("\nTrain DBH summary:")
    print(train_df["DBH_CM"].describe())

    print("\nValidation DBH summary:")
    print(val_df["DBH_CM"].describe())

    train_ds = TreeDBHDataset(
        train_df,
        image_size=args.image_size,
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=True,
        use_log1p=args.use_log1p,
        use_species=args.use_species,
        species_dropout=args.species_dropout,
        species_unknown_idx=species_to_idx[SPECIES_UNKNOWN] if species_to_idx else 0,
    )

    val_ds = TreeDBHDataset(
        val_df,
        image_size=args.image_size,
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=False,
        use_log1p=args.use_log1p,
        use_species=args.use_species,
        species_dropout=0.0,
        species_unknown_idx=species_to_idx[SPECIES_UNKNOWN] if species_to_idx else 0,
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

    if args.use_species:
        model = build_resnet_with_species(
            backbone=args.backbone,
            in_channels=in_channels,
            device=device,
            num_species=len(species_to_idx),
            species_emb_dim=args.species_emb_dim,
            dropout_rate=args.dropout_rate,
        )
    else:
        model = build_resnet(
            backbone=args.backbone,
            in_channels=in_channels,
            device=device,
            dropout_rate=args.dropout_rate,
        )

    if args.criterion == "smoothl1":
        criterion = nn.SmoothL1Loss(beta=1.0)
    else:
            criterion = nn.HuberLoss(delta=5.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "plateau": 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
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
            gamma=0.1,
        )
    else:
        scheduler = None

    models_root = out_root / "tree_dbh_models"
    models_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_name is not None:
        run_name = f"{args.run_name}_{timestamp}"
    else:
        species_tag = "_species" if args.use_species else ""
        run_name = (
            f"dbh_{args.backbone}_"
            f"{args.input_mode}_{args.image_source}{species_tag}_{timestamp}"
        )

    run_dir = models_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pth"
    last_model_path = run_dir / "last_model.pth"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"
    species_mapping_path = run_dir / "species_to_idx.json"

    if args.use_species:
        with open(species_mapping_path, "w") as f:
            json.dump(species_to_idx, f, indent=4)

    config = {
        "task": "dbh_regression",
        "target": "DBH_CM",
        "dataset_dir": str(dataset_dir),
        "backbone": args.backbone,
        "in_channels": in_channels,
        "input_mode": args.input_mode,
        "image_source": args.image_source,
        "use_depth": args.input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"},
        "use_sam": args.input_mode in {"rgb_sam", "rgb_sam_depth", "rgb_sam3", "rgb_sam3_depth"},
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "num_workers": args.num_workers,
        "loss": args.criterion if hasattr(args, "criterion") else "huber",
        "device": str(device),
        "use_log1p": args.use_log1p,
        "use_species": args.use_species,
        "species_emb_dim": args.species_emb_dim if args.use_species else None,
        "species_min_count": args.species_min_count if args.use_species else None,
        "species_dropout": args.species_dropout if args.use_species else None,
        "num_species": len(species_to_idx) if args.use_species else None,
        "species_mapping_path": str(species_mapping_path) if args.use_species else None,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_r2": [],
        "val_r2": [],
        "lr": [],
    }

    best_val_mae = float("inf")
    best_epoch = -1

    print(f"\nSaving training run to: {run_dir}")

    for epoch in range(args.epochs):
        train_loss, train_mae, train_rmse, train_r2, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            use_log1p=args.use_log1p,
        )

        val_loss, val_mae, val_rmse, val_r2, val_preds, val_targets = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            use_log1p=args.use_log1p,
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_mae)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_mae"].append(float(train_mae))
        history["val_mae"].append(float(val_mae))
        history["train_rmse"].append(float(train_rmse))
        history["val_rmse"].append(float(val_rmse))
        history["train_r2"].append(float(train_r2))
        history["val_r2"].append(float(val_r2))
        history["lr"].append(float(current_lr))

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_MAE={train_mae:.2f}cm "
            f"train_RMSE={train_rmse:.2f}cm | "
            f"val_loss={val_loss:.4f} val_MAE={val_mae:.2f}cm "
            f"val_RMSE={val_rmse:.2f}cm val_R2={val_r2:.4f} | "
            f"lr={current_lr:.2e}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "backbone": args.backbone,
            "scheduler": args.scheduler if hasattr(args, "scheduler") else "none",
            "in_channels": in_channels,
            "input_mode": args.input_mode,
            "image_source": args.image_source,
            "image_size": args.image_size,
            "target": "DBH_CM",
            "use_species": args.use_species,
            "num_species": len(species_to_idx) if args.use_species else None,
            "species_emb_dim": args.species_emb_dim if args.use_species else None,
            "val_loss": float(val_loss),
            "val_mae": float(val_mae),
            "val_rmse": float(val_rmse),
            "val_r2": float(val_r2),
        }

        torch.save(checkpoint, last_model_path)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            torch.save(checkpoint, best_model_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_mae_cm": float(best_val_mae),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_val_mae_cm": float(history["val_mae"][-1]),
        "final_val_rmse_cm": float(history["val_rmse"][-1]),
        "final_val_r2": float(history["val_r2"][-1]),
        "n_train_images": int(len(train_df)),
        "n_val_images": int(len(val_df)),
        "n_train_trees": int(train_df["ID"].nunique()),
        "n_val_trees": int(val_df["ID"].nunique()),
        "use_species": args.use_species,
        "num_species": len(species_to_idx) if args.use_species else None,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


    np.save(run_dir / "val_targets_dbh_cm.npy", val_targets)
    np.save(run_dir / "val_preds_dbh_cm.npy", val_preds)
    np.save(run_dir / "val_labels.npy", val_targets)
    np.save(run_dir / "val_preds.npy", val_preds)

    pred_cols = ["ID", "IMAGE_ID", "DBH_CM"]
    if args.use_species:
        pred_cols += ["TREE_SPECIES_RAW", "TREE_SPECIES_LABEL", "TREE_SPECIES_IDX"]

    pred_df = val_df[pred_cols].copy()
    pred_df["PRED_DBH_CM"] = val_preds
    pred_df["ABS_ERROR_DBH_CM"] = (pred_df["PRED_DBH_CM"] - pred_df["DBH_CM"]).abs()
    pred_df.to_csv(run_dir / "val_predictions.csv", index=False)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val MAE: {best_val_mae:.2f} cm")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved last model to: {last_model_path}")
    print(f"Saved validation predictions to: {run_dir / 'val_predictions.csv'}")


if __name__ == "__main__":
    main()