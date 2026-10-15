#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download pretrained LeXT-Net checkpoints from a single Zenodo ZIP archive.

Zenodo release layout
---------------------
The project publishes one archive named::

    LeXT_Net_release_weights.zip

The archive is intentionally flat (no required subdirectories) and contains
release-named checkpoints such as::

    ptbxl__EGV3__leXT-netv2__fold1.pth
    chapman__EGV1__leXT-netv2__fold5.pth
    cpsc2018__QSI__leXT-netv2__fold3.pth

and, preferably, ``weights_manifest.csv`` with SHA-256 hashes.

This script downloads the archive once, keeps it in a local cache, and extracts
only the checkpoint subset requested by the user into::

    results/weights/<dataset>/<strategy>/<model>/fold<k>.pth

No model training is performed.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZENODO_DOI = "10.5281/zenodo.23123820"
ZENODO_RECORD_ID = ZENODO_DOI.split("zenodo.", 1)[1]
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ARCHIVE_NAME = "LeXT_Net_release_weights.zip"

CHECKPOINT_RE = re.compile(
    r"^(ptbxl|chapman|cpsc2018)__(EGV1|EGV2|EGV3|QSI)__"
    r"(leXT-netv1|leXT-netv1s|leXT-netv2|leXT-netv2s)__fold([1-5])\.pth$"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Download/extract LeXT-Net pretrained weights from the Zenodo release archive"
    )
    p.add_argument("--dataset", default="all",
                   choices=["all", "ptbxl", "chapman", "cpsc2018"])
    p.add_argument("--strategy", default="all",
                   choices=["all", "EGV1", "EGV2", "EGV3", "QSI"])
    p.add_argument("--model", default="leXT-netv2",
                   choices=["all", "leXT-netv1", "leXT-netv1s", "leXT-netv2", "leXT-netv2s"])
    p.add_argument("--fold", default="all", choices=["all", "1", "2", "3", "4", "5"])
    p.add_argument("--list", action="store_true",
                   help="List checkpoints contained in the published archive")
    p.add_argument("--force", action="store_true",
                   help="Re-download the archive and overwrite extracted checkpoints")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip Zenodo archive checksum and manifest SHA-256 verification")
    p.add_argument("--cache-dir", default=str(ROOT / "results" / "weights" / "_cache"),
                   help="Local cache directory for the Zenodo ZIP archive")
    p.add_argument("--out-dir", default=str(ROOT / "results" / "weights"),
                   help="Root directory for extracted checkpoints")
    return p.parse_args()


def query_zenodo_record():
    try:
        with urllib.request.urlopen(ZENODO_API, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        raise RuntimeError(
            f"Could not query Zenodo record {ZENODO_DOI}: {e}\n"
            f"Visit https://doi.org/{ZENODO_DOI} manually."
        ) from e


def locate_archive(record):
    files = record.get("files", [])
    # Prefer the exact public filename.
    for f in files:
        if f.get("key") == ARCHIVE_NAME:
            return f
    # Tolerate a version suffix if maintainers later rename the archive.
    candidates = [
        f for f in files
        if f.get("key", "").lower().startswith("lext_net")
        and "weight" in f.get("key", "").lower()
        and f.get("key", "").lower().endswith(".zip")
    ]
    if len(candidates) == 1:
        return candidates[0]
    available = ", ".join(f.get("key", "") for f in files) or "<none>"
    raise RuntimeError(
        f"Could not uniquely locate {ARCHIVE_NAME} in the Zenodo record.\n"
        f"Available files: {available}"
    )


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_remote_checksum(path, checksum):
    if not checksum:
        return True, "Zenodo did not expose a checksum"
    if ":" not in checksum:
        return True, f"unrecognized checksum format: {checksum}"
    alg, expected = checksum.split(":", 1)
    alg = alg.lower()
    try:
        actual = digest(path, alg)
    except ValueError:
        return True, f"unsupported checksum algorithm: {alg}"
    return actual.lower() == expected.lower(), f"{alg}:{actual}"


def archive_download_url(file_meta):
    links = file_meta.get("links", {})
    # Zenodo API responses commonly expose 'content'; older records may use 'self'.
    return links.get("content") or links.get("self")


def ensure_archive(file_meta, cache_dir, force=False, verify=True):
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / file_meta.get("key", ARCHIVE_NAME)
    checksum = file_meta.get("checksum", "")

    if archive_path.exists() and not force:
        if verify:
            ok, msg = verify_remote_checksum(archive_path, checksum)
            if ok:
                print(f"Archive present/verified: {archive_path}")
                return archive_path
            print(f"Cached archive checksum mismatch ({msg}); re-downloading.")
        else:
            print(f"Archive present: {archive_path}")
            return archive_path

    url = archive_download_url(file_meta)
    if not url:
        raise RuntimeError("Zenodo archive metadata does not contain a downloadable URL.")

    tmp = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"Downloading archive: {file_meta.get('key', ARCHIVE_NAME)}")
    print(f"Destination       : {archive_path}")
    try:
        urllib.request.urlretrieve(url, tmp)
        if verify:
            ok, msg = verify_remote_checksum(tmp, checksum)
            if not ok:
                raise RuntimeError(f"Zenodo archive checksum verification failed ({msg})")
        tmp.replace(archive_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return archive_path


def checkpoint_info(member_name):
    # Accept both flat archives and accidental directory prefixes by using basename.
    name = Path(member_name).name
    m = CHECKPOINT_RE.match(name)
    if not m:
        return None
    dataset, strategy, model, fold = m.groups()
    return {
        "member": member_name,
        "filename": name,
        "dataset": dataset,
        "strategy": strategy,
        "model": model,
        "fold": fold,
    }


def list_checkpoints(zf):
    items = []
    seen = set()
    for member in zf.namelist():
        info = checkpoint_info(member)
        if info is None:
            continue
        if info["filename"] in seen:
            raise RuntimeError(f"Duplicate checkpoint filename in archive: {info['filename']}")
        seen.add(info["filename"])
        items.append(info)
    return sorted(items, key=lambda x: (
        x["dataset"], x["strategy"], x["model"], int(x["fold"])
    ))


def filter_checkpoints(items, args):
    return [
        x for x in items
        if (args.dataset == "all" or x["dataset"] == args.dataset)
        and (args.strategy == "all" or x["strategy"] == args.strategy)
        and (args.model == "all" or x["model"] == args.model)
        and (args.fold == "all" or x["fold"] == args.fold)
    ]


def load_manifest(zf):
    candidates = [n for n in zf.namelist() if Path(n).name == "weights_manifest.csv"]
    if not candidates:
        return {}
    if len(candidates) > 1:
        raise RuntimeError("Multiple weights_manifest.csv files found in archive.")
    raw = zf.read(candidates[0]).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw)))
    out = {}
    for row in rows:
        filename = Path(row.get("filename", "")).name
        if filename:
            out[filename] = row
    return out


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def verify_member_bytes(filename, data, manifest):
    row = manifest.get(filename)
    if row is None:
        return True, "not listed in weights_manifest.csv"
    expected = (row.get("sha256") or "").strip().lower()
    if not expected:
        return True, "manifest has no SHA-256 value"
    actual = sha256_bytes(data)
    return actual == expected, actual


def extract_selected(zf, selected, manifest, out_root, force=False, verify=True):
    ok_count = 0
    for item in selected:
        dest = (out_root / item["dataset"] / item["strategy"] /
                item["model"] / f"fold{item['fold']}.pth")

        if dest.exists() and not force:
            if verify and item["filename"] in manifest:
                expected = (manifest[item["filename"]].get("sha256") or "").strip().lower()
                if expected and digest(dest, "sha256").lower() == expected:
                    print(f"present/verified: {dest}")
                    ok_count += 1
                    continue
                print(f"checksum mismatch; replacing: {dest}")
            else:
                print(f"present: {dest}")
                ok_count += 1
                continue

        data = zf.read(item["member"])
        if verify:
            valid, msg = verify_member_bytes(item["filename"], data, manifest)
            if not valid:
                raise RuntimeError(
                    f"SHA-256 mismatch for {item['filename']} (actual {msg})"
                )

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        tmp.replace(dest)
        print(f"extracted: {item['filename']} -> {dest}")
        ok_count += 1
    return ok_count


def main():
    args = parse_args()
    record = query_zenodo_record()
    archive_meta = locate_archive(record)
    archive = ensure_archive(
        archive_meta,
        Path(args.cache_dir),
        force=args.force,
        verify=not args.no_verify,
    )

    try:
        with zipfile.ZipFile(archive, "r") as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"Corrupt ZIP member detected: {bad}")

            items = list_checkpoints(zf)
            if not items:
                raise RuntimeError("No release-named .pth checkpoints were found inside the archive.")

            selected = filter_checkpoints(items, args)
            if not selected:
                print("No matching checkpoint files were found in the published archive.")
                raise SystemExit(1)

            manifest = load_manifest(zf)

            if args.list:
                print(f"Archive: {archive_meta.get('key', ARCHIVE_NAME)}")
                print(f"Available matching checkpoints ({len(selected)}):")
                for x in selected:
                    mark = " [manifest]" if x["filename"] in manifest else ""
                    print(
                        f"  {x['dataset']}/{x['strategy']}/{x['model']}/"
                        f"fold{x['fold']}.pth{mark}"
                    )
                print(f"\nTotal checkpoints in archive: {len(items)}")
                return

            ok = extract_selected(
                zf, selected, manifest, Path(args.out_dir),
                force=args.force, verify=not args.no_verify
            )
            print(f"\nExtracted/present: {ok}/{len(selected)}")
            print(f"Cached archive   : {archive}")
            if ok != len(selected):
                raise SystemExit(2)

    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {archive}") from e


if __name__ == "__main__":
    main()
