#!/usr/bin/env python3
"""download_data.py — fetch the genuine public source dataset for Slate Forensics.

Source dataset
--------------
Online Retail II — real transactions of a UK-based online giftware retailer,
2009-12-01 .. 2011-12-09, donated to the UCI Machine Learning Repository
(dataset id 502) under the Creative Commons Attribution 4.0 International
license (CC BY 4.0).

    dataset page : https://archive.ics.uci.edu/dataset/502/online+retail+ii
    direct file  : https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
    metadata API : https://archive.ics.uci.edu/api/dataset?id=502

What this script does
---------------------
1. Downloads the official zip archive into data/raw/ (no credentials, no
   paid access — a plain HTTPS GET against UCI's static file host).
2. Extracts the contained .xlsx workbook into data/raw/.
3. Fetches the UCI metadata API for dataset id 502 and CROSS-CHECKS the
   license and DOI at download time. If the license reported by the source
   is not CC BY 4.0, the script ABORTS instead of writing provenance —
   provenance is never fabricated, only recorded from the live source.
4. Writes:
      data/raw/uci_api_metadata.json   (verbatim API response — evidence)
      data/raw/checksums.txt           (sha256 of every downloaded file)
      data/raw/source_license.txt      (license + attribution statement)
      data/source_metadata.json        (machine-readable provenance)

If the network is unavailable the script fails with a clear message and
writes nothing; no placeholder or invented data is ever produced.

Usage:
    python download_data.py [--root <project_root>] [--force]
"""

import argparse
import hashlib
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

DATASET_ID = 502
DATASET_NAME = "Online Retail II"
PAGE_URL = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
ZIP_URLS = [  # primary + fallback spellings used by UCI's static host
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
    "https://archive.ics.uci.edu/static/public/502/data.zip",
]
API_URL = "https://archive.ics.uci.edu/api/dataset?id=502"
LICENSE_NAME = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
# Accepted spellings of the license in UCI metadata / page HTML.
LICENSE_PATTERNS = [r"CC\s*BY\s*4\.0", r"Attribution\s*4\.0"]

UA = {"User-Agent": "eris-challenge-builder/1.0 (dataset provenance fetch)"}
TIMEOUT = 120


def _ssl_context() -> ssl.SSLContext:
    """Default trust store; fall back to certifi's CA bundle when the local
    Python install ships without root certificates (common on macOS)."""
    ctx = ssl.create_default_context()
    try:  # probe whether the default store can actually verify anything
        if not ctx.cert_store_stats()["x509_ca"]:
            raise ValueError("empty trust store")
        return ctx
    except Exception:  # noqa: BLE001
        import certifi  # available in the Kaggle image; optional elsewhere
        return ssl.create_default_context(cafile=certifi.where())


_CTX = _ssl_context()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=_CTX) as resp:
        return resp.read()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_license_evidence(api_json: dict, page_html: str | None) -> str:
    """Return a human-readable evidence string proving the license, or raise."""
    blob = json.dumps(api_json, ensure_ascii=False)
    for pat in LICENSE_PATTERNS:
        m = re.search(pat, blob, flags=re.IGNORECASE)
        if m:
            return f"UCI metadata API response contains license string {m.group(0)!r}"
    if page_html:
        for pat in LICENSE_PATTERNS:
            m = re.search(pat, page_html, flags=re.IGNORECASE)
            if m:
                return f"UCI dataset page HTML contains license string {m.group(0)!r}"
    raise RuntimeError(
        "Could not confirm the CC BY 4.0 license from the live UCI source. "
        "Refusing to write provenance. Inspect data/raw/uci_api_metadata.json "
        "manually and re-run, or select another dataset."
    )


def find_doi(api_json: dict, page_html: str | None) -> str | None:
    """Extract the dataset DOI from live metadata; None if absent (never invented)."""
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() == "doi" and isinstance(v, str) and v.strip():
                    yield v.strip()
                else:
                    yield from walk(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from walk(it)

    for doi in walk(api_json):
        return doi
    if page_html:
        m = re.search(r"10\.24432/[A-Z0-9]+", page_html, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the zip already exists")
    args = parser.parse_args()

    raw = args.root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "online_retail_ii.zip"

    # ------------------------------------------------------------------ zip
    if zip_path.exists() and zip_path.stat().st_size > 0 and not args.force:
        print(f"[skip] {zip_path.name} already present "
              f"({zip_path.stat().st_size/1e6:.1f} MB); use --force to re-fetch")
    else:
        last_err: Exception | None = None
        for url in ZIP_URLS:
            try:
                print(f"[get ] {url}")
                blob = fetch(url)
                if len(blob) < 1_000_000:
                    raise RuntimeError(f"suspiciously small response ({len(blob)} bytes)")
                zip_path.write_bytes(blob)
                print(f"[ok  ] wrote {zip_path.name} ({len(blob)/1e6:.1f} MB)")
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 — report every fallback
                print(f"[warn] {url} failed: {exc}")
                last_err = exc
        if last_err is not None:
            sys.exit(
                "FATAL: could not download the source dataset from any known "
                "official URL. No data was fabricated. Check network access and "
                f"retry; the dataset page is {PAGE_URL}"
            )

    # -------------------------------------------------------------- extract
    with zipfile.ZipFile(zip_path) as zf:
        xlsx_members = [m for m in zf.namelist() if m.lower().endswith(".xlsx")]
        if not xlsx_members:
            sys.exit("FATAL: no .xlsx member inside the downloaded zip — "
                     "unexpected archive layout, aborting.")
        extracted = []
        for member in sorted(xlsx_members):
            target = raw / Path(member).name.replace(" ", "_").lower()
            with zf.open(member) as src:
                target.write_bytes(src.read())
            extracted.append(target)
            print(f"[ok  ] extracted {member!r} -> {target.name}")

    # --------------------------- canonical CSV export (platform upload form)
    # The Eris upload form accepts csv/json/parquet/archives but not xlsx, so
    # the canonical raw file is a lossless CSV export of the workbook (same
    # rows, same columns; timestamps rendered as YYYY-MM-DD HH:MM:SS and
    # customer ids as integers). prepare.py prefers this CSV and falls back
    # to the xlsx; both yield byte-identical prepared outputs.
    import pandas as pd  # Kaggle-image library; used only for the export
    xl = pd.ExcelFile(raw / "online_retail_ii.xlsx")
    frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
    full = pd.concat(frames, ignore_index=True)
    full["Customer ID"] = pd.to_numeric(full["Customer ID"],
                                        errors="coerce").astype("Int64")
    full["InvoiceDate"] = full["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    csv_path = raw / "retail_transactions_2009_2011.csv"
    full.to_csv(csv_path, index=False)
    print(f"[ok  ] exported {csv_path.name} ({len(full):,} rows)")

    # ------------------------------------------- live license / DOI evidence
    api_json: dict = {}
    page_html: str | None = None
    try:
        api_json = json.loads(fetch(API_URL).decode("utf-8"))
        (raw / "uci_api_metadata.json").write_text(
            json.dumps(api_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("[ok  ] saved uci_api_metadata.json")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] metadata API failed ({exc}); falling back to page HTML")
    try:
        page_html = fetch(PAGE_URL).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] dataset page fetch failed ({exc})")

    evidence = find_license_evidence(api_json, page_html)
    print(f"[ok  ] license confirmed live: {evidence}")
    doi = find_doi(api_json, page_html)
    print(f"[ok  ] DOI from live source: {doi!r}")

    # Official citation exactly as published on the UCI dataset page.
    citation = (
        "Chen, D. (2012). Online Retail II [Dataset]. "
        "UCI Machine Learning Repository."
        + (f" https://doi.org/{doi}." if doi else "")
    )

    # ----------------------------------------------------------- checksums
    files = sorted(p for p in raw.iterdir() if p.is_file() and p.suffix != ".txt")
    checksums = {p.name: sha256_file(p) for p in files}
    with open(raw / "checksums.txt", "w", encoding="utf-8") as f:
        for name, digest in checksums.items():
            f.write(f"{digest}  {name}\n")
    print(f"[ok  ] checksums.txt ({len(checksums)} files)")

    # -------------------------------------------------------- license note
    (raw / "source_license.txt").write_text(
        f"{DATASET_NAME}\n"
        f"Source: {PAGE_URL}\n"
        f"License: {LICENSE_NAME}\n"
        f"License URL: {LICENSE_URL}\n"
        f"License evidence at download time: {evidence}\n"
        f"Attribution: {citation}\n"
        f"CC BY 4.0 permits redistribution, adaptation and commercial use "
        f"provided attribution is given. This challenge redistributes a "
        f"cleaned/derived form with attribution; see dataset_description.md.\n",
        encoding="utf-8",
    )

    # ------------------------------------------------- machine-readable prov
    meta = {
        "dataset_name": DATASET_NAME,
        "source_url": PAGE_URL,
        "download_urls": ZIP_URLS[:1],
        "license_name": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "license_evidence": evidence,
        "citation": citation,
        "accession_or_doi": doi,
        "uci_dataset_id": DATASET_ID,
        "access_date": date.today().isoformat(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "downloaded_files": [p.name for p in files],
        "sha256_checksums": checksums,
        "transformation_summary": (
            "prepare.py keeps every real transaction field untouched and adds a "
            "deterministic synthetic overlay (seed 20260705): a hidden 'You may "
            "also like' recommendation policy is replayed at fixed anchor dates "
            "over real customer histories to emit 10-item slates; a seeded "
            "minority of slates is corrupted by one of three realistic failure "
            "modes (popularity fallback, price-band shift, stale index). Public "
            "files: cleaned transactions, catalog, labeled train slates with "
            "healthy references, unlabeled test slates, sample submission. "
            "Private: per-test-slate flag, failure mode, healthy slate and "
            "emitted slate for consistency grading. Full details in prepare.py."
        ),
    }
    out = args.root / "data" / "source_metadata.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[ok  ] wrote {out.relative_to(args.root)}")
    print("\nDownload + provenance complete.")


if __name__ == "__main__":
    main()
