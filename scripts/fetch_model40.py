from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = "https://vectors.nlpl.eu/repository/20/40.zip"
EXPECTED_MODEL_SIZE = 1_651_062_031
EXPECTED_VOCAB = 4_027_169
EXPECTED_DIM = 100

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARCHIVE_PATH = MODELS_DIR / "40.zip"
MODEL_PATH = MODELS_DIR / "model.bin"
MANIFEST_PATH = MODELS_DIR / "model40_local_manifest.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temp = destination.with_suffix(destination.suffix + ".part")
    if temp.exists():
        temp.unlink()
    with urllib.request.urlopen(url) as response, temp.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    temp.replace(destination)


def extract_model(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if Path(name).name == "model.bin"]
        if len(members) != 1:
            raise RuntimeError(f"Expected exactly one model.bin in archive, found {members}")
        member = members[0]
        temp = destination.with_suffix(destination.suffix + ".part")
        if temp.exists():
            temp.unlink()
        with archive.open(member) as source, temp.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        temp.replace(destination)


def validate_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != EXPECTED_MODEL_SIZE:
        raise RuntimeError(f"Unexpected model.bin size: {size}; expected {EXPECTED_MODEL_SIZE}")
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8", errors="strict").strip().split()
    if len(header) < 2:
        raise RuntimeError("Invalid Word2Vec header")
    vocab, dim = int(header[0]), int(header[1])
    if vocab != EXPECTED_VOCAB or dim != EXPECTED_DIM:
        raise RuntimeError(
            f"Unexpected Word2Vec header: vocab={vocab}, dim={dim}; "
            f"expected vocab={EXPECTED_VOCAB}, dim={EXPECTED_DIM}"
        )
    return {
        "model_url": MODEL_URL,
        "model_path": str(path.relative_to(ROOT)),
        "file_size_bytes": size,
        "sha256": sha256_file(path),
        "vocabulary_size": vocab,
        "dimensions": dim,
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        print(f"Model already present: {MODEL_PATH}")
    else:
        print(f"Downloading NLPL Model 40 from {MODEL_URL}")
        download(MODEL_URL, ARCHIVE_PATH)
        print("Extracting model.bin")
        extract_model(ARCHIVE_PATH, MODEL_PATH)

    manifest = validate_model(MODEL_PATH)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()

    print("Model 40 ready.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
