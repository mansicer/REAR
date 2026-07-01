#!/usr/bin/env python3
"""Reconstruct the MMHal-Bench image set from the committed metadata.

The 96 MMHal-Bench images (~170 MB) are *not* stored in this repository. Instead
we keep only the lightweight metadata file ``response_template.json`` (image URLs,
questions, ground-truth answers) and rebuild ``images/`` on demand from the
original source URLs.

Each record in ``response_template.json`` carries an ``image_src`` URL; the image
is saved to ``images/<basename(image_src)>`` — exactly the path that
``utils/data.py`` looks up when building the multimodal request. The raw bytes are
written unmodified, so the on-disk format matches what the evaluation pipeline
base64-encodes.

Usage (run from anywhere; paths are resolved relative to this file):

    python data/mmhal-bench/download_images.py            # download missing images
    python data/mmhal-bench/download_images.py --force    # re-download everything
    python data/mmhal-bench/download_images.py --workers 8

If some URLs have rotted, fetch the upstream archive instead and unzip its
``images/`` here:  https://huggingface.co/datasets/Shengcao1006/MMHal-Bench
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "response_template.json")
IMAGE_DIR = os.path.join(HERE, "images")
UA = "Mozilla/5.0 (compatible; REAR-eval-kit/1.0; +MMHal-Bench image fetch)"


def download_one(url: str, dest: str, retries: int = 3, timeout: int = 30) -> None:
    """Download ``url`` to ``dest``, retrying on transient failures."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise ValueError("empty response body")
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001 - report and retry any fetch error
            last_err = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download images even if they already exist")
    parser.add_argument("--workers", type=int, default=4, help="number of parallel download workers")
    args = parser.parse_args()

    with open(TEMPLATE, "r") as f:
        records = json.load(f)

    os.makedirs(IMAGE_DIR, exist_ok=True)

    # De-duplicate by destination filename (basename of the source URL).
    todo = {}
    for rec in records:
        url = rec["image_src"]
        dest = os.path.join(IMAGE_DIR, os.path.basename(url))
        if not args.force and os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        todo[dest] = url

    total = len(records)
    have = total - len(todo)
    print(f"MMHal-Bench: {total} images total, {have} already present, {len(todo)} to download.")
    if not todo:
        print("Nothing to do.")
        return 0

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, url, dest): dest for dest, url in todo.items()}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            dest = futures[fut]
            name = os.path.basename(dest)
            try:
                fut.result()
                print(f"[{i}/{len(todo)}] ok   {name}")
            except Exception as e:  # noqa: BLE001
                failures.append((name, str(e)))
                print(f"[{i}/{len(todo)}] FAIL {name}: {e}")

    if failures:
        print(f"\n{len(failures)} image(s) failed to download:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        print(
            "\nSome Flickr/OpenImages URLs may have expired. Fetch the upstream archive and\n"
            "unzip its images/ into this directory instead:\n"
            "  https://huggingface.co/datasets/Shengcao1006/MMHal-Bench"
        )
        return 1

    print(f"\nDone. All images are in {IMAGE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
