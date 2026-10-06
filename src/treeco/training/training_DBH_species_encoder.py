#!/usr/bin/env python3
from __future__ import annotations
import wandb

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path
from treeco.image_models.encoder import TreeCoEncoder

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
        sample_weight_col: str | None = None,
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
        self.sample_weight_col = sample_weight_col

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
        if (
            self.sample_weight_col is not None
            and self.sample_weight_col not in self.df.columns
        ):
            raise ValueError(
                f"Sample weight column "
                f"{self.sample_weight_col!r} is missing."
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

        # =========================================================
        # RGB
        # =========================================================

        rgb = Image.open(
            row[self.rgb_col]
        ).convert("RGB")

        single_channels = []

        # =========================================================
        # Optional image modalities
        # =========================================================

        if self.use_sam:
            single_channels.append(
                self._load_single_channel_image(
                    row["SAM_LOGITS_PATH"]
                )
            )

        if self.use_sam3:
            single_channels.append(
                self._load_single_channel_image(
                    row["SAM3_MASK_PATH"]
                )
            )

        if self.use_depth:
            single_channels.append(
                self._load_single_channel_image(
                    row["DEPTH_PATH"]
                )
            )

        # =========================================================
        # Shared geometric transforms
        # =========================================================

        rgb, single_channels = (
            self._apply_shared_geometric_transforms(
                rgb,
                single_channels,
            )
        )

        # =========================================================
        # Convert to tensors
        # =========================================================

        rgb_tensor = self._rgb_to_tensor(rgb)

        channels = [rgb_tensor]

        for ch in single_channels:
            channels.append(
                self._single_to_tensor(
                    ch,
                    dtype=rgb_tensor.dtype,
                )
            )

        x = torch.cat(
            channels,
            dim=0,
        )

        # =========================================================
        # DBH target
        # =========================================================

        dbh_cm = float(
            row["DBH_CM"]
        )

        if self.use_log1p:
            target = np.log1p(dbh_cm)
        else:
            target = dbh_cm

        y = torch.tensor(
            target,
            dtype=torch.float32,
        )

        # =========================================================
        # Optional training-loss weight
        # =========================================================

        sample_weight = None

        if self.sample_weight_col is not None:

            sample_weight = torch.tensor(
                float(
                    row[self.sample_weight_col]
                ),
                dtype=torch.float32,
            )

        # =========================================================
        # Optional species metadata
        # =========================================================

        if self.use_species:

            species_idx = int(
                row["TREE_SPECIES_IDX"]
            )

            # During training, occasionally replace species
            # with Unknown so that the model does not become
            # too dependent on species metadata.
            if (
                self.train
                and self.species_dropout > 0
            ):
                if random.random() < self.species_dropout:
                    species_idx = (
                        self.species_unknown_idx
                    )

            species_idx = torch.tensor(
                species_idx,
                dtype=torch.long,
            )

            # Species + training-loss weighting
            if sample_weight is not None:
                return (
                    x,
                    species_idx,
                    y,
                    sample_weight,
                )

            # Species only
            return (
                x,
                species_idx,
                y,
            )

        # =========================================================
        # No species metadata
        # =========================================================

        # Training-loss weighting only
        if sample_weight is not None:
            return (
                x,
                y,
                sample_weight,
            )

        # Original behaviour
        return (
            x,
            y,
        )

class TreeCoMultimodalResNetEncoder(nn.Module):
    """
    TreeCo multimodal image encoder.

    Each modality is encoded independently before fusion:

        RGB   -> TreeCoEncoder
        SAM   -> TreeCoEncoder
        Depth -> TreeCoEncoder

    The fused representation is then passed through the pretrained
    residual blocks of a ResNet.

    The custom encoders + fusion block replace the normal
    ResNet stem (conv1, bn1, relu, maxpool).
    """

    def __init__(
        self,
        backbone: str,
        input_mode: str,
    ):
        super().__init__()

        self.input_mode = input_mode

        # -----------------------------------------------------
        # Which modalities are present?
        # -----------------------------------------------------

        self.use_mask = input_mode in {
            "rgb_sam",
            "rgb_sam_depth",
            "rgb_sam3",
            "rgb_sam3_depth",
        }

        self.use_depth = input_mode in {
            "rgb_depth",
            "rgb_sam_depth",
            "rgb_sam3_depth",
        }

        # -----------------------------------------------------
        # Individual modality encoders
        # -----------------------------------------------------

        # RGB:
        # [B, 3, 224, 224]
        # ->
        # [B, 32, 56, 56]

        self.rgb_encoder = TreeCoEncoder(
            in_channels=3,
            hidden_channels=32,
            out_channels=32,
            first_kernel_size=7,
        )

        fusion_channels = 32

        # SAM / SAM3:
        # [B, 1, 224, 224]
        # ->
        # [B, 8, 56, 56]

        if self.use_mask:
            self.mask_encoder = TreeCoEncoder(
                in_channels=1,
                hidden_channels=8,
                out_channels=8,
                first_kernel_size=5,
            )

            fusion_channels += 8
        else:
            self.mask_encoder = None

        # Depth:
        # [B, 1, 224, 224]
        # ->
        # [B, 16, 56, 56]

        if self.use_depth:
            self.depth_encoder = TreeCoEncoder(
                in_channels=1,
                hidden_channels=16,
                out_channels=16,
                first_kernel_size=5,
            )

            fusion_channels += 16
        else:
            self.depth_encoder = None

        # -----------------------------------------------------
        # Fusion
        # -----------------------------------------------------

        # Example for rgb_sam3_depth:
        #
        # 32 + 8 + 16 = 56 channels
        #
        # -> 64 channels
        #
        # This produces the same dimensionality expected by
        # ResNet layer1.

        self.fusion_encoder = nn.Sequential(
            nn.Conv2d(
                fusion_channels,
                64,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # -----------------------------------------------------
        # Pretrained ResNet
        # -----------------------------------------------------

        backbone = backbone.lower()

        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "resnet101": models.ResNet101_Weights.DEFAULT,
        }

        if backbone not in weights_map:
            raise ValueError(
                f"Unsupported backbone: {backbone}"
            )

        resnet = getattr(
            models,
            backbone,
        )(
            weights=weights_map[backbone]
        )

        # We skip the original ResNet stem:
        #
        # conv1
        # bn1
        # relu
        # maxpool
        #
        # because our modality encoders + fusion replace it.

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.image_feat_dim = resnet.fc.in_features


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # -----------------------------------------------------
        # Expected input layouts
        #
        # rgb:
        #   RGB
        #
        # rgb_depth:
        #   RGB | depth
        #
        # rgb_sam / rgb_sam3:
        #   RGB | mask
        #
        # rgb_sam_depth / rgb_sam3_depth:
        #   RGB | mask | depth
        # -----------------------------------------------------

        expected_channels = (
            3
            + int(self.use_mask)
            + int(self.use_depth)
        )

        if x.shape[1] != expected_channels:
            raise ValueError(
                f"{self.input_mode} expects "
                f"{expected_channels} channels, "
                f"but received {x.shape[1]}"
            )

        encoded = []

        # -----------------------------------------------------
        # RGB
        # -----------------------------------------------------

        rgb = x[:, 0:3, :, :]

        rgb = self.rgb_encoder(
            rgb
        )

        encoded.append(rgb)

        channel_idx = 3

        # -----------------------------------------------------
        # SAM / SAM3
        # -----------------------------------------------------

        if self.use_mask:

            mask = x[
                :,
                channel_idx:channel_idx + 1,
                :,
                :
            ]

            channel_idx += 1

            mask = self.mask_encoder(
                mask
            )

            encoded.append(mask)

        # -----------------------------------------------------
        # Depth
        # -----------------------------------------------------

        if self.use_depth:

            depth = x[
                :,
                channel_idx:channel_idx + 1,
                :,
                :
            ]

            depth = self.depth_encoder(
                depth
            )

            encoded.append(depth)

        # -----------------------------------------------------
        # Concatenate modalities
        # -----------------------------------------------------

        x = torch.cat(
            encoded,
            dim=1,
        )

        # -----------------------------------------------------
        # Fusion
        # -----------------------------------------------------

        x = self.fusion_encoder(
            x
        )

        # -----------------------------------------------------
        # Pretrained ResNet residual blocks
        # -----------------------------------------------------

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # -----------------------------------------------------
        # Global image features
        # -----------------------------------------------------

        x = self.pool(x)

        x = torch.flatten(
            x,
            1,
        )

        return x


class ResNetDBH(nn.Module):

    def __init__(
        self,
        image_encoder: nn.Module,
        image_feat_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.image_encoder = image_encoder

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(
                image_feat_dim,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        image_feat = self.image_encoder(
            x
        )

        return self.head(
            image_feat
        )


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


def build_standard_resnet_encoder(
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



def build_standard_resnet_with_species(
    backbone: str,
    in_channels: int,
    device: torch.device,
    num_species: int,
    species_emb_dim: int = 16,
    dropout_rate: float = 0.1,
) -> nn.Module:

    image_encoder, image_feat_dim = build_standard_resnet_encoder(
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

def build_standard_resnet(
    backbone: str,
    in_channels: int,
    device: torch.device,
    dropout_rate: float = 0.1,
) -> nn.Module:

    image_encoder, image_feat_dim = build_standard_resnet_encoder(
        backbone=backbone,
        in_channels=in_channels,
        device=device,
    )

    model = ResNetDBH(
        image_encoder=image_encoder,
        image_feat_dim=image_feat_dim,
        dropout_rate=dropout_rate,
    )

    return model.to(device)

def build_encoded_resnet(
    backbone: str,
    input_mode: str,
    device: torch.device,
    dropout_rate: float = 0.1,
) -> nn.Module:

    image_encoder = TreeCoMultimodalResNetEncoder(
        backbone=backbone,
        input_mode=input_mode,
    )

    model = ResNetDBH(
        image_encoder=image_encoder,
        image_feat_dim=image_encoder.image_feat_dim,
        dropout_rate=dropout_rate,
    )

    return model.to(device)


def build_encoded_resnet_with_species(
    backbone: str,
    input_mode: str,
    device: torch.device,
    num_species: int,
    species_emb_dim: int = 16,
    dropout_rate: float = 0.1,
) -> nn.Module:

    image_encoder = TreeCoMultimodalResNetEncoder(
        backbone=backbone,
        input_mode=input_mode,
    )

    model = ResNetDBHWithSpecies(
        image_encoder=image_encoder,
        image_feat_dim=image_encoder.image_feat_dim,
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
    use_species: bool = False,
    use_sample_weights: bool = False,
):
    train_mode = optimizer is not None

    if train_mode:
        model.train()
    else:
        model.eval()

    all_preds_raw = []
    all_targets_raw = []

    total_loss_numerator = 0.0
    total_loss_denominator = 0.0

    for batch in loader:

        sample_weight = None
        species_idx = None

        # =====================================================
        # Unpack batch
        # =====================================================

        if use_species and use_sample_weights:

            x, species_idx, y, sample_weight = batch

        elif use_species:

            x, species_idx, y = batch

        elif use_sample_weights:

            x, y, sample_weight = batch

        else:

            x, y = batch

        # =====================================================
        # Move to device
        # =====================================================

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        if species_idx is not None:
            species_idx = species_idx.to(
                device,
                non_blocking=True,
            )

        if sample_weight is not None:
            sample_weight = sample_weight.to(
                device,
                non_blocking=True,
            )

        # =====================================================
        # Zero gradients only during training
        # =====================================================

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        # =====================================================
        # Forward + loss
        # =====================================================

        with torch.set_grad_enabled(train_mode):

            if species_idx is None:
                preds = model(x).squeeze(1)
            else:
                preds = model(
                    x,
                    species_idx,
                ).squeeze(1)

            # criterion uses reduction="none"
            per_sample_loss = criterion(
                preds,
                y,
            )

            if per_sample_loss.ndim > 1:
                per_sample_loss = (
                    per_sample_loss
                    .view(per_sample_loss.size(0), -1)
                    .mean(dim=1)
                )

            # =================================================
            # Optional training-loss weighting
            # =================================================

            if sample_weight is not None:

                sample_weight = sample_weight.to(
                    dtype=per_sample_loss.dtype,
                )

                weighted_sum = (
                    per_sample_loss
                    * sample_weight
                ).sum()

                weight_sum = (
                    sample_weight
                    .sum()
                    .clamp_min(1e-8)
                )

                loss = (
                    weighted_sum
                    / weight_sum
                )

                total_loss_numerator += (
                    weighted_sum
                    .detach()
                    .item()
                )

                total_loss_denominator += (
                    weight_sum
                    .detach()
                    .item()
                )

            else:

                loss = per_sample_loss.mean()

                total_loss_numerator += (
                    per_sample_loss
                    .detach()
                    .sum()
                    .item()
                )

                total_loss_denominator += (
                    per_sample_loss.numel()
                )

            # =================================================
            # Optimisation
            # =================================================

            if train_mode:
                loss.backward()
                optimizer.step()

        # =====================================================
        # Save predictions for ordinary metrics
        # =====================================================

        all_preds_raw.extend(
            preds.detach().cpu().numpy()
        )

        all_targets_raw.extend(
            y.detach().cpu().numpy()
        )

    # =========================================================
    # Epoch loss
    # =========================================================

    avg_loss = (
        total_loss_numerator
        / max(total_loss_denominator, 1e-8)
    )

    all_preds_raw = np.asarray(
        all_preds_raw,
        dtype=np.float32,
    )

    all_targets_raw = np.asarray(
        all_targets_raw,
        dtype=np.float32,
    )

    # =========================================================
    # R2 in target space
    # =========================================================

    try:
        r2_target = r2_score(
            all_targets_raw,
            all_preds_raw,
        )
    except Exception:
        r2_target = np.nan

    # =========================================================
    # Convert back to DBH cm
    # =========================================================

    if use_log1p:

        all_preds = np.expm1(
            all_preds_raw
        )

        all_targets = np.expm1(
            all_targets_raw
        )

        all_preds = np.clip(
            all_preds,
            0,
            None,
        )

        all_targets = np.clip(
            all_targets,
            0,
            None,
        )

    else:

        all_preds = all_preds_raw
        all_targets = all_targets_raw

    # =========================================================
    # Ordinary unweighted metrics in DBH cm
    # =========================================================

    mae = mean_absolute_error(
        all_targets,
        all_preds,
    )

    rmse = mean_squared_error(
        all_targets,
        all_preds,
    ) ** 0.5

    try:
        r2_cm = r2_score(
            all_targets,
            all_preds,
        )
    except Exception:
        r2_cm = np.nan

    return (
        avg_loss,
        mae,
        rmse,
        r2_cm,
        r2_target,
        all_preds,
        all_targets,
    )



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



def parse_dbh_weight_bins(spec: str) -> np.ndarray:
    """
    Parse comma-separated DBH bin edges.

    Example:
        "0,20,40,60,80,100,inf"
    """
    parts = [p.strip().lower() for p in str(spec).split(",") if p.strip()]

    if len(parts) < 2:
        raise ValueError("--dbh_weight_bins must contain at least two edges.")

    edges = []
    for part in parts:
        if part in {"inf", "+inf", "infinity", "+infinity"}:
            edges.append(np.inf)
        elif part in {"-inf", "-infinity"}:
            edges.append(-np.inf)
        else:
            edges.append(float(part))

    edges = np.asarray(edges, dtype=float)

    if not np.all(np.diff(edges) > 0):
        raise ValueError(
            "--dbh_weight_bins must be strictly increasing, "
            f"got {edges.tolist()}"
        )

    return edges


def add_dbh_training_weights(
    df: pd.DataFrame,
    bin_edges: np.ndarray,
    power: float = 0.5,
    max_weight: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add a DBH-frequency weight using UNIQUE TRAINING TREES, not images.

    For DBH bin b:

        raw_weight_b = (max_bin_count / bin_count_b) ** power

    With power=0.5 this is square-root inverse-frequency weighting.
    Weights are clipped to max_weight.

    Returns
    -------
    weighted_df:
        Original image-level dataframe with DBH_BIN,
        N_IMAGES_PER_TREE and DBH_TRAIN_WEIGHT.

    summary:
        One row per occupied DBH bin with counts and weights.
    """
    if power < 0:
        raise ValueError("dbh_weight_power must be >= 0.")

    if max_weight <= 0:
        raise ValueError("dbh_weight_max must be > 0.")

    out = df.copy()

    # One row per independent tree.
    tree_info = (
        out.groupby("ID", as_index=False)
        .agg(
            DBH_CM=("DBH_CM", "median"),
            N_IMAGES_PER_TREE=("ID", "size"),
        )
    )

    tree_info["DBH_BIN"] = pd.cut(
        tree_info["DBH_CM"],
        bins=bin_edges,
        right=False,
        include_lowest=True,
    )

    if tree_info["DBH_BIN"].isna().any():
        bad = tree_info.loc[
            tree_info["DBH_BIN"].isna(),
            ["ID", "DBH_CM"],
        ]
        raise ValueError(
            "Some training-tree DBH values fall outside --dbh_weight_bins. "
            f"Examples:\n{bad.head()}"
        )

    counts = (
        tree_info.groupby("DBH_BIN", observed=True)
        .size()
        .rename("N_TREES")
    )

    max_count = float(counts.max())

    weights = (
        (max_count / counts.astype(float)) ** float(power)
    ).clip(upper=float(max_weight))

    weights.name = "DBH_TRAIN_WEIGHT"

    summary = pd.concat(
        [counts, weights],
        axis=1,
    ).reset_index()

    tree_info = tree_info.merge(
        summary[["DBH_BIN", "DBH_TRAIN_WEIGHT"]],
        on="DBH_BIN",
        how="left",
        validate="many_to_one",
    )

    # Make the bin label easy to save/debug later.
    tree_info["DBH_BIN"] = tree_info["DBH_BIN"].astype(str)

    out = out.merge(
        tree_info[
            [
                "ID",
                "DBH_BIN",
                "N_IMAGES_PER_TREE",
                "DBH_TRAIN_WEIGHT",
            ]
        ],
        on="ID",
        how="left",
        validate="many_to_one",
    )

    out["DBH_TRAIN_WEIGHT"] = pd.to_numeric(
        out["DBH_TRAIN_WEIGHT"],
        errors="raise",
    ).astype(np.float32)

    return out, summary


def add_tree_balance_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1 / number-of-images-for-tree.

    This stops trees with many photographs from automatically contributing
    proportionally more total loss than trees with only one photograph.
    """
    out = df.copy()

    if "N_IMAGES_PER_TREE" not in out.columns:
        counts = out.groupby("ID")["ID"].transform("size")
        out["N_IMAGES_PER_TREE"] = counts.astype(int)

    out["TREE_BALANCE_WEIGHT"] = (
        1.0
        / out["N_IMAGES_PER_TREE"].astype(float)
    ).astype(np.float32)

    return out


def combine_training_weights(
    df: pd.DataFrame,
    use_dbh_weights: bool,
    use_tree_balance: bool,
    final_weight_max: float | None = None,
) -> pd.DataFrame:
    """
    Combine enabled loss-weight components multiplicatively:

        FINAL = DBH_WEIGHT * SAM3_WEIGHT * TREE_BALANCE_WEIGHT

    Disabled components contribute 1.0.

    The final image weights are normalised to mean 1 so the overall loss
    scale stays comparable across experiments.
    """
    out = df.copy()

    final_weight = np.ones(len(out), dtype=np.float64)

    if use_dbh_weights:
        if "DBH_TRAIN_WEIGHT" not in out.columns:
            raise ValueError("DBH weighting enabled but DBH_TRAIN_WEIGHT is missing.")
        final_weight *= out["DBH_TRAIN_WEIGHT"].to_numpy(dtype=np.float64)

    if use_tree_balance:
        if "TREE_BALANCE_WEIGHT" not in out.columns:
            raise ValueError(
                "Tree balancing enabled but TREE_BALANCE_WEIGHT is missing."
            )
        final_weight *= out["TREE_BALANCE_WEIGHT"].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(final_weight)) or np.any(final_weight <= 0):
        raise ValueError("Final training weights must all be finite and > 0.")

    # Keep average loss scale similar to the unweighted experiment.
    final_weight /= final_weight.mean()

    # Optional strict safety cap for very rare DBH + one-image combinations.
    # We do not renormalise after clipping; run_epoch divides by sum(weights),
    # so an overall multiplicative scale does not change the weighted loss.
    if final_weight_max is not None:
        final_weight_max = float(final_weight_max)
        if final_weight_max <= 0:
            raise ValueError("--final_weight_max must be > 0.")
        final_weight = np.minimum(final_weight, final_weight_max)

    out["TRAIN_WEIGHT"] = final_weight.astype(np.float32)

    return out


def print_training_weight_diagnostics(
    df: pd.DataFrame,
    use_dbh_weights: bool,
    use_tree_balance: bool,
) -> None:
    print("\n" + "=" * 60)
    print("TRAINING WEIGHT DIAGNOSTICS")
    print("=" * 60)

    print(f"DBH weighting:       {use_dbh_weights}")
    print(f"Tree balancing:      {use_tree_balance}")

    for col in [
        "DBH_TRAIN_WEIGHT",
        "TREE_BALANCE_WEIGHT",
        "TRAIN_WEIGHT",
    ]:
        if col in df.columns:
            print(f"\n{col}")
            print(df[col].describe())

    if "TRAIN_WEIGHT" in df.columns:
        tree_totals = (
            df.groupby("ID", as_index=False)
            .agg(
                DBH_CM=("DBH_CM", "median"),
                N_IMAGES=("ID", "size"),
                TOTAL_TRAIN_WEIGHT=("TRAIN_WEIGHT", "sum"),
                MEAN_TRAIN_WEIGHT=("TRAIN_WEIGHT", "mean"),
            )
        )

        print("\nPer-tree total training-weight summary")
        print(tree_totals["TOTAL_TRAIN_WEIGHT"].describe())

        print("\nHighest-weight training trees")
        print(
            tree_totals.sort_values(
                "TOTAL_TRAIN_WEIGHT",
                ascending=False,
            ).head(10).to_string(index=False)
        )


def make_dbh_weight_bins(
    dbh_values,
    bin_size: float,
) -> np.ndarray:
    """
    Create equal-width DBH bins starting at 0.

    Example
    -------
    If bin_size = 20 and max DBH = 127 cm:

        [0, 20, 40, 60, 80, 100, 120, 140]

    The final edge is guaranteed to be above the maximum DBH.
    """

    if bin_size <= 0:
        raise ValueError(
            "dbh_bin_size must be > 0."
        )

    dbh_values = pd.to_numeric(
        pd.Series(dbh_values),
        errors="coerce",
    ).dropna()

    if len(dbh_values) == 0:
        raise ValueError(
            "Cannot construct DBH bins: no valid DBH values."
        )

    max_dbh = float(
        dbh_values.max()
    )

    # Need an upper edge strictly above the largest value
    upper = (
        np.floor(max_dbh / bin_size)
        + 1
    ) * bin_size

    bins = np.arange(
        0,
        upper + bin_size,
        bin_size,
        dtype=float,
    )

    return bins


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
    "--wandb_project",
    type=str,
    default="treeco-dbh",
    )

    ap.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
    )

    ap.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
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


    # ------------------------------------------------------------
    # Optional training-loss weighting
    # ------------------------------------------------------------
    ap.add_argument(
        "--w_dbh",
        action="store_true",
        help=(
            "Enable DBH-frequency loss weighting based only on the "
            "TRAINING trees."
        ),
    )

    ap.add_argument(
        "--w_sam3",
        action="store_true",
        help=(
            "Enable SAM3 confidence loss weighting. Requires "
            "--sam3_weights <config.json>."
        ),
    )

    ap.add_argument(
        "--w_tree_balance",
        action="store_true",
        help=(
            "Multiply each image by 1/N_images_for_tree so trees with "
            "many photographs do not dominate the loss."
        ),
    )

    ap.add_argument(
        "--dbh_bin_size",
        type=float,
        default=20.0,
        help=(
            "Width of DBH bins in cm used for DBH loss weighting. "
            "For example, --dbh_bin_size 20 creates bins "
            "0-20, 20-40, 40-60, etc."
        ),
    )

    ap.add_argument(
        "--dbh_weight_power",
        type=float,
        default=0.5,
        help=(
            "Inverse-frequency exponent for DBH weighting. "
            "0.5 = square-root inverse frequency; 1.0 = full inverse frequency."
        ),
    )

    ap.add_argument(
        "--dbh_weight_max",
        type=float,
        default=3.0,
        help="Maximum DBH-frequency weight before combining components.",
    )

    ap.add_argument(
        "--final_weight_max",
        type=float,
        default=None,
        help=(
            "Optional cap on the final combined per-image weight after "
            "normalisation. Example: 6.0. Default: no final cap."
        ),
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

    ap.add_argument(
        "--use_encoder",
        action="store_true",
        help="Use separate modality encoders before ResNet fusion.",
    )

    ap.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help=(
            "Stop training if validation MAE does not improve for this "
            "many consecutive epochs. Set to 0 to disable."
        ),
    )

    ap.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.05,
        help=(
            "Minimum improvement in validation MAE [cm] required "
            "to reset early-stopping patience."
        ),
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

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=vars(args),
        name=args.run_name,
    )

    cfg = wandb.config

    sweep_params = [
        "dataset_name",
        "dataset_path",
        "out_dir",

        "backbone",
        "input_mode",
        "image_source",
        "use_encoder",
        "use_species",
        "use_log1p",

        "species_emb_dim",
        "species_min_count",
        "species_dropout",

        "image_size",
        "batch_size",
        "dropout_rate",
        "epochs",
        "lr",
        "weight_decay",
        "criterion",
        "scheduler",
        "val_size",
        "random_state",
        "w_dbh",
        "w_tree_balance",
        "dbh_weight_bins",
        "dbh_weight_power",
        "dbh_weight_max",
        "final_weight_max",
        "num_workers",
    ]

    for name in sweep_params:
        if name in cfg:
            setattr(args, name, cfg[name])

    # =========================================================
    # Weighting configuration
    # =========================================================

    use_dbh_weights = bool(args.w_dbh)
    use_tree_balance = bool(args.w_tree_balance)

    use_sample_weights = any(
        [
            use_dbh_weights,
            use_tree_balance,
        ]
    )

    # These will be created later, AFTER train_df exists
    dbh_weight_bin_edges = None
    dbh_weight_summary = None


    print("\n" + "=" * 60)
    print("LOSS WEIGHTING SETUP")
    print("=" * 60)

    print(
        f"DBH weighting:  "
        f"{use_dbh_weights}"
    )

    print(
        f"Tree balancing: "
        f"{use_tree_balance}"
    )

    if use_dbh_weights:

        print(
            f"DBH bin size:    "
            f"{args.dbh_bin_size:.1f} cm"
        )

        print(
            f"DBH power:       "
            f"{args.dbh_weight_power:.3f}"
        )

        print(
            f"DBH max weight:  "
            f"{args.dbh_weight_max:.3f}"
        )


    if args.final_weight_max is not None:

        print(
            f"Final weight cap: "
            f"{args.final_weight_max:.3f}"
        )

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

    stratify_bins = make_regression_stratification_bins(tree_df)

    train_tree_ids, val_tree_ids = train_test_split(
        tree_df["ID"],
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=stratify_bins,
    )

    train_df = df[df["ID"].isin(train_tree_ids)].copy().reset_index(drop=True)
    val_df = df[df["ID"].isin(val_tree_ids)].copy().reset_index(drop=True)

    # =========================================================
    # Build TRAINING loss weights AFTER the train/val split.
    # Nothing from validation is used to estimate these weights.
    # =========================================================

    dbh_weight_summary = None


    # ---------------------------------------------------------
    # 1. Construct DBH bin edges from TRAINING data only
    # ---------------------------------------------------------

    if use_dbh_weights:

        dbh_weight_bin_edges = make_dbh_weight_bins(
            train_df["DBH_CM"],
            bin_size=args.dbh_bin_size,
        )

        print("\nDBH weighting bins:")
        print(dbh_weight_bin_edges)

        print(
            f"DBH bin size: "
            f"{args.dbh_bin_size:.1f} cm"
        )


    # ---------------------------------------------------------
    # 2. DBH-frequency weighting
    # ---------------------------------------------------------

    if use_dbh_weights:

        train_df, dbh_weight_summary = (
            add_dbh_training_weights(
                train_df,
                bin_edges=dbh_weight_bin_edges,
                power=args.dbh_weight_power,
                max_weight=args.dbh_weight_max,
            )
        )

        print(
            "\nDBH weighting by TRAINING-tree frequency"
        )

        print(
            "----------------------------------------"
        )

        print(
            dbh_weight_summary.to_string(
                index=False
            )
        )


    # ---------------------------------------------------------
    # 3. Tree balancing
    # ---------------------------------------------------------

    if use_tree_balance:

        train_df = add_tree_balance_weights(
            train_df
        )


    # ---------------------------------------------------------
    # 4. Combine enabled weighting components
    # ---------------------------------------------------------

    if use_sample_weights:

        train_df = combine_training_weights(
            train_df,
            use_dbh_weights=use_dbh_weights,
            use_tree_balance=use_tree_balance,
            final_weight_max=args.final_weight_max,
        )


        print_training_weight_diagnostics(
            train_df,
            use_dbh_weights=use_dbh_weights,
            use_tree_balance=use_tree_balance,
        )

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
        species_unknown_idx=(
            species_to_idx[SPECIES_UNKNOWN]
            if species_to_idx
            else 0
        ),
        sample_weight_col=(
            "TRAIN_WEIGHT"
            if use_sample_weights
            else None
        ),
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
        species_unknown_idx=(
            species_to_idx[SPECIES_UNKNOWN]
            if species_to_idx
            else 0
        ),
        sample_weight_col=None,
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

    # =========================================================
    # Build model
    # =========================================================

    if args.use_encoder:

        print("\nUsing modality encoders:")
        print("  RGB   -> TreeCoEncoder")
        print("  SAM   -> TreeCoEncoder (if enabled)")
        print("  Depth -> TreeCoEncoder (if enabled)")
        print("  -> Fusion -> ResNet")

        if args.use_species:

            model = build_encoded_resnet_with_species(
                backbone=args.backbone,
                input_mode=args.input_mode,
                device=device,
                num_species=len(species_to_idx),
                species_emb_dim=args.species_emb_dim,
                dropout_rate=args.dropout_rate,
            )

        else:

            model = build_encoded_resnet(
                backbone=args.backbone,
                input_mode=args.input_mode,
                device=device,
                dropout_rate=args.dropout_rate,
            )

    else:

        print("\nUsing standard ResNet input:")
        print(f"  Input channels: {in_channels}")
        print("  -> ResNet")

        if args.use_species:

            model = build_standard_resnet_with_species(
                backbone=args.backbone,
                in_channels=in_channels,
                device=device,
                num_species=len(species_to_idx),
                species_emb_dim=args.species_emb_dim,
                dropout_rate=args.dropout_rate,
            )

        else:

            model = build_standard_resnet(
                backbone=args.backbone,
                in_channels=in_channels,
                device=device,
                dropout_rate=args.dropout_rate,
            )

    if args.criterion == "smoothl1":

            criterion = nn.SmoothL1Loss(
                beta=1.0,
                reduction="none",
            )

    else:

            criterion = nn.HuberLoss(
                delta=5.0,
                reduction="none",
            )

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
        encoder_tag = "_encoder" if args.use_encoder else "_noEncoder"
        species_tag = "_species" if args.use_species else ""
        dbh_weight_tag = "_wDBH" if use_dbh_weights else ""
        tree_weight_tag = "_wTree" if use_tree_balance else ""

        run_name = (
            f"dbh_{args.backbone}_"
            f"{args.input_mode}_{args.image_source}"
            f"{encoder_tag}{species_tag}"
            f"{dbh_weight_tag}{tree_weight_tag}_"
            f"{timestamp}"
        )

    wandb_run.name = run_name

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

    if use_sample_weights:
        weight_cols = [
            c for c in [
                "ID",
                "IMAGE_ID",
                "DBH_CM",
                "DBH_BIN",
                "N_IMAGES_PER_TREE",
                "DBH_TRAIN_WEIGHT",
                "SAM3_TRAIN_WEIGHT",
                "TREE_BALANCE_WEIGHT",
                "TRAIN_WEIGHT",
            ]
            if c in train_df.columns
        ]

        train_df[weight_cols].to_csv(
            run_dir / "training_sample_weights.csv",
            index=False,
        )

        if dbh_weight_summary is not None:
            dbh_weight_summary.to_csv(
                run_dir / "dbh_weight_summary.csv",
                index=False,
            )

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
        "use_encoder": args.use_encoder,
        "encoder_config": {
                            "rgb_out_channels": 32,
                            "mask_out_channels": 8,
                            "depth_out_channels": 16,
                            "fusion_out_channels": 64,
                        } if args.use_encoder else None,
        "weighting_enabled": use_sample_weights,
        "w_dbh": use_dbh_weights,
        "w_tree_balance": use_tree_balance,
        "dbh_weight_bins": dbh_weight_bin_edges.tolist() if dbh_weight_bin_edges is not None else None,
        "dbh_weight_power": float(args.dbh_weight_power),
        "dbh_weight_max": float(args.dbh_weight_max),
        "final_weight_max": (
            float(args.final_weight_max)
            if args.final_weight_max is not None
            else None
        ),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
    }

    wandb.config.update(
    config,
    allow_val_change=True,
    )

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

            "train_r2_cm": [],
            "val_r2_cm": [],

            "train_r2_target": [],
            "val_r2_target": [],

            "lr": [],
        }

    best_val_mae = float("inf")
    best_val_rmse = np.nan

    best_epoch = -1
    # =========================================================
    # Early stopping state
    # =========================================================

    epochs_without_improvement = 0
    early_stop_best_mae = float("inf")
    stopped_early = False

    best_val_r2_cm = np.nan
    best_val_r2_target = np.nan

    best_val_preds = None
    best_val_targets = None
    
    print(f"\nSaving training run to: {run_dir}")

    for epoch in range(args.epochs):
        (
            train_loss,
            train_mae,
            train_rmse,
            train_r2_cm,
            train_r2_target,
            _,
            _,
        ) = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            use_log1p=args.use_log1p,
            use_species=args.use_species,
            use_sample_weights=use_sample_weights,
        )

        (
            val_loss,
            val_mae,
            val_rmse,
            val_r2_cm,
            val_r2_target,
            val_preds,
            val_targets,
        ) = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            use_log1p=args.use_log1p,
            use_species=args.use_species,
            use_sample_weights=False,
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

        # Backwards-compatible aliases for plotting
        history["train_r2"].append(float(train_r2_cm))
        history["val_r2"].append(float(val_r2_cm))

        # Explicit R2 metrics
        history["train_r2_cm"].append(float(train_r2_cm))
        history["val_r2_cm"].append(float(val_r2_cm))

        history["train_r2_target"].append(float(train_r2_target))
        history["val_r2_target"].append(float(val_r2_target))

        history["lr"].append(float(current_lr))

        wandb.log({
        "epoch": epoch + 1,

        "train/loss": train_loss,
        "train/mae_cm": train_mae,
        "train/rmse_cm": train_rmse,
        "train/r2_cm": train_r2_cm,
        "train/r2_target": train_r2_target,

        "val/loss": val_loss,
        "val/mae_cm": val_mae,
        "val/rmse_cm": val_rmse,
        "val/r2_cm": val_r2_cm,
        "val/r2_target": val_r2_target,

        "learning_rate": current_lr,
        })


        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_MAE={train_mae:.2f}cm "
            f"train_RMSE={train_rmse:.2f}cm "
            f"train_R2={train_r2_cm:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_MAE={val_mae:.2f}cm "
            f"val_RMSE={val_rmse:.2f}cm "
            f"val_R2={val_r2_cm:.4f} "
            f"val_R2_target={val_r2_target:.4f} | "
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

            "val_r2_cm": float(val_r2_cm),
            "val_r2_target": float(val_r2_target),

            "use_log1p": args.use_log1p,
            "use_encoder": args.use_encoder,
            "weighting_enabled": use_sample_weights,
            "w_dbh": use_dbh_weights,
            "w_tree_balance": use_tree_balance,
            "dbh_weight_bins": dbh_weight_bin_edges.tolist() if dbh_weight_bin_edges is not None else None,
            "dbh_weight_power": float(args.dbh_weight_power),
            "dbh_weight_max": float(args.dbh_weight_max),
            "final_weight_max": (
                float(args.final_weight_max)
                if args.final_weight_max is not None
                else None
            ),
        }

        torch.save(checkpoint, last_model_path)

        # =========================================================
        # Save best model
        # =========================================================

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            best_val_r2_cm = val_r2_cm
            best_val_r2_target = val_r2_target
            best_val_preds = val_preds.copy()
            best_val_targets = val_targets.copy()

            wandb.run.summary["best_val_mae_cm"] = val_mae
            wandb.run.summary["best_val_rmse_cm"] = val_rmse
            wandb.run.summary["best_val_r2_cm"] = val_r2_cm
            wandb.run.summary["best_epoch"] = epoch + 1

            torch.save(checkpoint, best_model_path)


        # =========================================================
        # Early stopping
        # =========================================================

        if args.early_stopping_patience > 0:

            significant_improvement = (
                val_mae < early_stop_best_mae - args.early_stopping_min_delta
            )

            if significant_improvement:
                early_stop_best_mae = val_mae
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Early stopping: {epochs_without_improvement}/"
                f"{args.early_stopping_patience}"
            )

            if epochs_without_improvement >= args.early_stopping_patience:
                stopped_early = True

                print("\n" + "=" * 60)
                print("EARLY STOPPING")
                print("=" * 60)
                print(
                    f"No validation MAE improvement of at least "
                    f"{args.early_stopping_min_delta:.3f} cm for "
                    f"{args.early_stopping_patience} epochs."
                )
                print(f"Best epoch: {best_epoch}")
                print(f"Best validation MAE: {best_val_mae:.2f} cm")
                print("=" * 60 + "\n")

                break

    if best_val_preds is not None:
        val_preds = best_val_preds
        val_targets = best_val_targets


    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)


    metrics = {
        "best_epoch": int(best_epoch),

        "best_val_mae_cm": float(best_val_mae),
        "best_val_rmse_cm": float(best_val_rmse),

        # Main R2 in actual DBH cm
        "best_val_r2_cm": float(best_val_r2_cm),
        "best_val_r2_target": float(best_val_r2_target),

        # Backwards-compatible name for plotting code
        "best_val_r2": float(best_val_r2_cm),

        "final_val_loss": float(history["val_loss"][-1]),
        "final_val_mae_cm": float(history["val_mae"][-1]),
        "final_val_rmse_cm": float(history["val_rmse"][-1]),

        "final_val_r2_cm": float(history["val_r2_cm"][-1]),
        "final_val_r2_target": float(history["val_r2_target"][-1]),

        # Backwards-compatible name for plotting code
        "final_val_r2": float(history["val_r2_cm"][-1]),

        "n_train_images": int(len(train_df)),
        "n_val_images": int(len(val_df)),
        "n_train_trees": int(train_df["ID"].nunique()),
        "n_val_trees": int(val_df["ID"].nunique()),

        "use_log1p": bool(args.use_log1p),
        "use_encoder": bool(args.use_encoder),
        "use_species": bool(args.use_species),
        "weighting_enabled": bool(use_sample_weights),
        "w_dbh": bool(use_dbh_weights),
        "w_tree_balance": bool(use_tree_balance),

        "num_species": (
            len(species_to_idx)
            if args.use_species
            else None
        ),
    }
    wandb.run.summary.update(metrics)

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

    wandb.log({
    "validation_predictions":
        wandb.Table(dataframe=pred_df)
    })
    

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val MAE: {best_val_mae:.2f} cm")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved last model to: {last_model_path}")
    print(f"Saved validation predictions to: {run_dir / 'val_predictions.csv'}")
    wandb.finish()

if __name__ == "__main__":
    main()