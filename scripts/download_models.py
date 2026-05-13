#!/usr/bin/env python3

from pathlib import Path

from treeco.image_models.download import download_model, list_models


def main():
    models = list_models()

    default_root = Path.home() / "treeco_models"

    root = input(f"Where should models be stored? [{default_root}]: ").strip()
    models_root = Path(root) if root else default_root

    print("\nAvailable models:")
    keys = list(models.keys())

    for i, key in enumerate(keys, start=1):
        print(f"[{i}] {models[key]['display_name']} ({key})")

    print(f"[{len(keys) + 1}] All")

    choice = input("\nChoose model(s), e.g. 1,3 or All: ").strip().lower()

    if choice in {"all", str(len(keys) + 1)}:
        selected = keys
    else:
        selected = []
        for part in choice.split(","):
            idx = int(part.strip()) - 1
            selected.append(keys[idx])

    for key in selected:
        print(f"\nDownloading {models[key]['display_name']}...")
        try:
            path = download_model(key, models_root)
            print(f"Saved to: {path}")
        except NotImplementedError as e:
            print(f"Skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()