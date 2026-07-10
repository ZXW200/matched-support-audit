"""Fetch fixed third-party repositories and the MediaPipe model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen


RESOURCES = {
    "YouTubePD-data": {
        "url": "https://github.com/samwli/YouTubePD-data.git",
        "commit": "43797386a65ffb58db53628b90ef8e8f35512e0d",
    },
    "UFNet": {
        "url": "https://github.com/ROC-HCI/UFNet.git",
        "commit": "5ece2c65ba184faccf6c8cdccdc03132427c464b",
    },
}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_repository(destination: Path, url: str, commit: str) -> None:
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--no-checkout", url, str(destination)])
    run(["git", "fetch", "origin", commit], cwd=destination)
    run(["git", "checkout", "--detach", commit], cwd=destination)
    observed = run(["git", "rev-parse", "HEAD"], cwd=destination)
    if observed.lower() != commit.lower():
        raise RuntimeError(f"Commit mismatch for {destination.name}: {observed}")


def fetch_model(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == MODEL_SHA256:
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".task") as temporary:
        temporary_path = Path(temporary.name)
        with urlopen(MODEL_URL) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
    observed = sha256(temporary_path)
    if observed != MODEL_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Face Landmarker hash mismatch: {observed}")
    temporary_path.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-dir", type=Path, default=Path("external"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("reproduction/third_party_manifest.json")
    )
    args = parser.parse_args()

    for name, resource in RESOURCES.items():
        fetch_repository(args.external_dir / name, resource["url"], resource["commit"])
    model_path = args.model_dir / "face_landmarker.task"
    fetch_model(model_path)

    manifest = {
        "repositories": RESOURCES,
        "face_landmarker": {
            "url": MODEL_URL,
            "sha256": sha256(model_path),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

