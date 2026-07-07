#!/usr/bin/env python3
"""download_data.py - fetch the genuine public source data for the
"Unfluke: Skill-vs-Luck Forensics for Systematic Trading Records" task.

Downloads one archive of real daily cryptocurrency OHLCV history from a
public Zenodo record (credential-free, CC BY 4.0), verifies its SHA-256
checksum, extracts the per-asset price CSVs into `data/raw/`, combines
them into the single canonical raw table `dataset/raw/prices.csv`, and
writes machine-readable provenance to `data/source_metadata.json`.

Nothing is fabricated: if the archive cannot be downloaded and no local
copy exists, the script fails with a clear error.

Usage:
    python download_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATASET_RAW = ROOT / "dataset" / "raw"
META_PATH = ROOT / "data" / "source_metadata.json"

SOURCE = {
    "dataset_name": "Binance cryptocurrencies historical daily data",
    "source_url": "https://zenodo.org/records/8187872",
    "download_url": (
        "https://zenodo.org/records/8187872/files/"
        "historical%20data%20and%20indicators.zip?download=1"
    ),
    "license_name": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
    "citation": (
        "Serhii Kanyhin (2023). Binance cryptocurrencies historical daily "
        "data [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8187872"
    ),
    "accession_or_doi": "10.5281/zenodo.8187872",
}

ARCHIVE_NAME = "source_daily_ohlcv.zip"
ARCHIVE_SHA256 = "ce9a73eb817d0d9c111f296f7dfdea1694d3bc8149d36375e6da25ce36fbed1f"
PRICE_PREFIX = "historical data/"  # only member files under this folder are used

TRANSFORMATION_SUMMARY = (
    "Only the per-asset daily OHLCV files under 'historical data/' inside "
    "the archive are used (the 'technical indicators/' folder is discarded). "
    "The per-asset CSVs are concatenated unchanged into the long-format "
    "table dataset/raw/prices.csv with columns symbol, open_time, "
    "close_time, open, high, low, close, volume_busd. All further "
    "transformations (asset anonymization, window selection, per-window "
    "normalization, deterministic +/-0.2% price jitter, strategy simulation "
    "and label construction) happen in prepare.py with fixed seeds."
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_archive(dest: Path) -> None:
    """Download the source archive to `dest` unless a verified copy exists."""
    if dest.exists() and sha256_of(dest) == ARCHIVE_SHA256:
        print(f"[skip] verified archive already present: {dest}")
        return
    print(f"[download] {SOURCE['download_url']}")
    try:
        req = urllib.request.Request(
            SOURCE["download_url"], headers={"User-Agent": "eris-task-build/1.0"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "ERROR: could not download the source archive and no verified "
            f"local copy exists at {dest}. Original error: {exc}. "
            "The data is public and credential-free; retry with internet "
            f"access or place the archive (sha256 {ARCHIVE_SHA256}) at that "
            "path manually."
        ) from exc
    got = sha256_of(dest)
    if got != ARCHIVE_SHA256:
        raise SystemExit(
            f"ERROR: downloaded archive checksum mismatch: expected "
            f"{ARCHIVE_SHA256}, got {got}. Refusing to continue."
        )
    print(f"[ok] archive verified: sha256 {got}")


def extract_prices(archive: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        members = [
            m
            for m in zf.namelist()
            if m.startswith(PRICE_PREFIX) and m.lower().endswith(".csv")
        ]
        if not members:
            raise SystemExit(
                "ERROR: archive does not contain the expected "
                f"'{PRICE_PREFIX}*.csv' members."
            )
        for m in sorted(members):
            name = Path(m).name
            target = out_dir / name
            with zf.open(m) as src:
                target.write_bytes(src.read())
            written.append(target)
    print(f"[ok] extracted {len(written)} per-asset price CSVs -> {out_dir}")
    return written


def build_prices_table(csv_files: list[Path], dest: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(csv_files):
        df = pd.read_csv(f, dtype=str)
        expected = [
            "symbol", "open_time", "close_time",
            "open", "high", "low", "close", "volume_busd",
        ]
        if list(df.columns) != expected:
            raise SystemExit(
                f"ERROR: unexpected columns in {f.name}: {list(df.columns)}"
            )
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["symbol", "open_time"], kind="mergesort")
    prices = prices.reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(dest, index=False)
    print(f"[ok] wrote {dest} ({len(prices)} rows, "
          f"{prices['symbol'].nunique()} assets)")
    return prices


def write_license_note(dest: Path) -> None:
    dest.write_text(
        "Source data license\n"
        "===================\n\n"
        f"Dataset: {SOURCE['dataset_name']}\n"
        f"Record:  {SOURCE['source_url']}\n"
        f"DOI:     {SOURCE['accession_or_doi']}\n"
        f"License: {SOURCE['license_name']}\n"
        f"         {SOURCE['license_url']}\n\n"
        f"Citation: {SOURCE['citation']}\n\n"
        "The upstream record distributes real daily OHLCV market history "
        "for 289 cryptocurrency assets (2021-06-01 to 2023-06-30). "
        "Attribution is preserved here and in data/source_metadata.json as "
        "required by CC BY 4.0.\n"
    )
    print(f"[ok] wrote {dest}")


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATASET_RAW.mkdir(parents=True, exist_ok=True)

    archive = DATA_RAW / ARCHIVE_NAME
    download_archive(archive)

    csv_files = extract_prices(archive, DATA_RAW / "historical_data")
    prices = build_prices_table(csv_files, DATASET_RAW / "prices.csv")
    write_license_note(DATASET_RAW / "source_license.txt")

    checksums = {
        ARCHIVE_NAME: sha256_of(archive),
        "dataset/raw/prices.csv": sha256_of(DATASET_RAW / "prices.csv"),
    }
    checksum_path = DATA_RAW / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(checksums.items()))
    )
    print(f"[ok] wrote {checksum_path}")

    meta = {
        **SOURCE,
        "access_date": str(date.today()),
        "downloaded_files": [ARCHIVE_NAME],
        "extracted_files": (
            f"{len(csv_files)} per-asset CSVs under data/raw/historical_data/"
        ),
        "sha256_checksums": checksums,
        "rows_in_combined_table": int(len(prices)),
        "assets_in_combined_table": int(prices["symbol"].nunique()),
        "date_range": [
            str(prices["open_time"].min()),
            str(prices["open_time"].max()),
        ],
        "transformation_summary": TRANSFORMATION_SUMMARY,
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[ok] wrote {META_PATH}")
    print("[done] raw data ready; next: python prepare.py")


if __name__ == "__main__":
    sys.exit(main())
