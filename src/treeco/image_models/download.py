import os
import subprocess
from getpass import getpass
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

from treeco.image_models.registry import MODEL_REGISTRY


# -------------------------
# PATH RESOLUTION
# -------------------------
def find_repo_root(start: Path | None = None) -> Path | None:
    start = start or Path(__file__).resolve()

    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path

    return None


def get_models_root() -> Path:
    env = os.getenv("TREECO_MODELS_DIR")

    if env:
        root = Path(env).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    repo_root = find_repo_root()
    if repo_root is not None:
        root = repo_root / "IMAGE_MODELS"
        root.mkdir(parents=True, exist_ok=True)
        return root

    root = Path.home() / "treeco_models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_model_path(model_key: str, models_root: str | Path | None = None) -> Path:
    if models_root is None:
        models_root = get_models_root()

    models_root = Path(models_root)
    info = MODEL_REGISTRY[model_key]

    output_dir = models_root / info["local_dir"]

    if "filename" in info:
        return output_dir / info["filename"]

    return output_dir


# -------------------------
# TOKEN HANDLING
# -------------------------
def get_hf_token_if_needed(info: dict) -> str | bool | None:
    """
    Return a Hugging Face token only for gated/private models.

    Priority:
        1. HF_TOKEN environment variable
        2. HUGGINGFACE_HUB_TOKEN environment variable
        3. Prompt user securely in terminal
        4. None
    """
    if not info.get("requires_hf_token", False):
        return None

    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if env_token:
        return env_token

    print()
    print("=" * 80)
    print("This model requires a Hugging Face access token.")
    print(f"Model: {info.get('display_name', info.get('repo_id', 'unknown'))}")
    print()
    print("You need to:")
    print("  1. Accept access to the gated model on Hugging Face.")
    print("  2. Create a Hugging Face access token.")
    print("  3. Paste the token below.")
    print()
    print("The token will not be shown while typing.")
    print("=" * 80)
    print()

    token = getpass("Hugging Face token: ").strip()

    if not token:
        raise RuntimeError(
            "No Hugging Face token provided. "
            "Cannot download this gated model."
        )

    return token


# -------------------------
# DOWNLOAD HELPERS
# -------------------------
class DownloadProgressBar(tqdm):
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_url_file(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return output_path

    with DownloadProgressBar(unit="B", unit_scale=True, desc=output_path.name) as p:
        urlretrieve(url, filename=output_path, reporthook=p.update_to)

    return output_path


def download_hf_snapshot(
    repo_id: str,
    output_dir: Path,
    token: str | bool | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        token=token,
    )

    return output_dir


def download_hf_file(
    repo_id: str,
    filename: str,
    output_dir: Path,
    token: str | bool | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return output_path

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=output_dir,
        token=token,
    )

    return Path(downloaded_path)


def download_git_repo(repo_url: str, output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        print(f"Already exists: {output_dir}")
        return output_dir

    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(output_dir)],
        check=True,
    )

    return output_dir


# -------------------------
# MAIN API
# -------------------------
def download_model(model_key: str, models_root: str | Path | None = None) -> Path:
    if models_root is None:
        models_root = get_models_root()

    models_root = Path(models_root)
    info = MODEL_REGISTRY[model_key]

    output_dir = models_root / info["local_dir"]

    hf_token = get_hf_token_if_needed(info)

    if info["type"] == "huggingface_snapshot":
        return download_hf_snapshot(
            repo_id=info["repo_id"],
            output_dir=output_dir,
            token=hf_token,
        )

    if info["type"] == "huggingface_file":
        return download_hf_file(
            repo_id=info["repo_id"],
            filename=info["filename"],
            output_dir=output_dir,
            token=hf_token,
        )

    if info["type"] == "url_file":
        return download_url_file(
            url=info["url"],
            output_path=output_dir / info["filename"],
        )

    if info["type"] == "git_clone":
        return download_git_repo(
            repo_url=info["repo_url"],
            output_dir=output_dir,
        )

    raise ValueError(f"Unknown model type: {info['type']}")


def list_models():
    return MODEL_REGISTRY