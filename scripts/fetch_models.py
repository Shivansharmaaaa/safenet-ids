"""Download the trained model bundles from the project's GitHub release, so a new sensor needs no training data.

usage: python -m scripts.fetch_models [--tag models-v1] [--force]
Files are checked against the SHA-256 sums published in the same release.
"""
from __future__ import annotations

import argparse
import hashlib
import ssl
import urllib.request

import certifi

from safenet.config import MODELS_DIR

REPO = "Shivansharmaaaa/safenet-ids"
FILES = ["live_bundle.joblib", "sequence_bundle.pt"]
_SSL = ssl.create_default_context(cafile=certifi.where())


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "safenet-sensor"})
    with urllib.request.urlopen(req, timeout=120, context=_SSL) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="release tag (default: latest release)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    base = (f"https://github.com/{REPO}/releases/download/{args.tag}" if args.tag
            else f"https://github.com/{REPO}/releases/latest/download")
    sums = dict(reversed(line.split()) for line in _get(f"{base}/models.sha256").decode().splitlines() if line.strip())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = MODELS_DIR / name
        if path.exists() and not args.force and hashlib.sha256(path.read_bytes()).hexdigest() == sums[name]:
            print(f"{name}: up to date")
            continue
        data = _get(f"{base}/{name}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != sums[name]:
            raise SystemExit(f"{name}: checksum mismatch ({digest} != {sums[name]})")
        path.write_bytes(data)
        print(f"{name}: downloaded {len(data) / 1e6:.1f} MB, sha256 ok")


if __name__ == "__main__":
    main()
