"""
Download NASA PCoE Li-ion Battery Aging Dataset.

Downloads the .mat files for cells B0005, B0006, B0007, B0018 from the
NASA PCoE data repository. These are the standard run-to-failure datasets
used in battery prognostics research.

Usage: python scripts/download_data.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

# NASA PCoE battery dataset URL
# The dataset is distributed as a single zip file containing all cell .mat files
DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

# Alternative mirrors (in case primary is down)
ALT_URLS = [
    "https://data.nasa.gov/download/7k4b-6rwy/application%2Fx-zip-compressed",
]

CELL_IDS = ["B0005", "B0006", "B0007", "B0018"]

def download_and_extract(data_dir: Path) -> None:
    """Download and extract NASA PCoE battery .mat files."""
    import urllib.request
    import urllib.error

    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already exist
    existing = [cid for cid in CELL_IDS if (data_dir / f"{cid}.mat").exists()]
    if len(existing) == len(CELL_IDS):
        print(f"All {len(CELL_IDS)} .mat files already present in {data_dir}")
        return

    if existing:
        print(f"Found {len(existing)}/{len(CELL_IDS)} files: {existing}")
        print("Downloading remaining...")

    urls = [DATASET_URL] + ALT_URLS
    downloaded = False

    for url in urls:
        print(f"Attempting download from:\n  {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = resp.headers.get("Content-Length")
                if total:
                    total = int(total)
                    print(f"  Size: {total / 1024 / 1024:.1f} MB")

                # Read into memory
                data = io.BytesIO()
                chunk_size = 1024 * 1024  # 1 MB
                downloaded_bytes = 0
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    data.write(chunk)
                    downloaded_bytes += len(chunk)
                    if total:
                        pct = downloaded_bytes / total * 100
                        print(f"\r  Downloaded: {downloaded_bytes/1024/1024:.1f} / "
                              f"{total/1024/1024:.1f} MB ({pct:.0f}%)", end="")
                    else:
                        print(f"\r  Downloaded: {downloaded_bytes/1024/1024:.1f} MB", end="")

                print()  # newline after progress

                # Extract .mat files from zip
                data.seek(0)
                with zipfile.ZipFile(data) as zf:
                    mat_files = [
                        name for name in zf.namelist()
                        if name.endswith(".mat") and
                        any(cid in name for cid in CELL_IDS)
                    ]
                    print(f"  Found {len(mat_files)} matching .mat files in archive")

                    for mat_file in mat_files:
                        # Extract just the filename
                        filename = Path(mat_file).name
                        target = data_dir / filename
                        if target.exists():
                            print(f"  Skipping {filename} (already exists)")
                            continue
                        print(f"  Extracting {filename}...")
                        with zf.open(mat_file) as src, open(target, "wb") as dst:
                            dst.write(src.read())

                downloaded = True
                break

        except (urllib.error.URLError, urllib.error.HTTPError, zipfile.BadZipFile) as e:
            print(f"  Failed: {e}")
            continue

    if not downloaded:
        print("\n[ERROR] Could not download from any mirror.")
        print("Please download manually from:")
        print("  https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository")
        print(f"  Place B0005.mat, B0006.mat, B0007.mat, B0018.mat in: {data_dir}")
        sys.exit(1)

    # Verify all files present
    missing = [cid for cid in CELL_IDS if not (data_dir / f"{cid}.mat").exists()]
    if missing:
        print(f"\n[WARNING] Still missing: {missing}")
        print("The archive may have different filenames or structure.")
        print(f"Please check {data_dir} and rename files if needed.")
    else:
        print(f"\n[OK] All {len(CELL_IDS)} .mat files present in {data_dir}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"
    download_and_extract(data_dir)


if __name__ == "__main__":
    main()
