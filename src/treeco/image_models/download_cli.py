from treeco.image_models.download import (
    download_model,
    list_models,
    get_models_root,
)


def main():
    models = list_models()

    default_root = get_models_root()

    root = input(f"Where should models be stored? [{default_root}]: ").strip()
    models_root = default_root if root == "" else root

    print("\nAvailable models:")
    keys = list(models.keys())

    for i, key in enumerate(keys, start=1):
        print(f"[{i}] {models[key]['display_name']} ({key})")

    print(f"[{len(keys) + 1}] All")

    choice = input("\nChoose model(s): ").strip().lower()

    if choice in {"all", str(len(keys) + 1)}:
        selected = keys
    else:
        selected = [keys[int(x.strip()) - 1] for x in choice.split(",")]

    for key in selected:
        print(f"\nDownloading {models[key]['display_name']}...")
        path = download_model(key, models_root)
        print(f"Saved to: {path}")

    print("\nDone.")