"""
Manual sanity check for the real-data loaders against the lab's actual
GitHub files. NOT part of the automated test suite (downloads multi-MB
files over the network) -- run this by hand:

    python scripts/manual_real_data_check.py

It downloads the two source files, runs both adapters, feeds their output
straight into DatasetBuilder.build(), and prints a summary so a human can
eyeball that the whole chain works end to end on real data.
"""

import tempfile
import urllib.request
from pathlib import Path

from flare_indexer import BinaryThresholdStrategy, DatasetBuilder
from flare_indexer.loaders import adapt_goes_catalog, build_image_index_from_filenames

IMAGE_LIST_URL = (
    "https://raw.githubusercontent.com/chetrajpandey/explainingFullDisk/"
    "main/labeling/data_labels/totalfiles_jpg_512.csv"
)
GOES_CATALOG_URL = (
    "https://raw.githubusercontent.com/chetrajpandey/explainingFullDisk/"
    "main/labeling/data_source/goes_flares_integrated.csv"
)


def _download(url: str, dest: Path) -> Path:
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        raw_image_list = _download(IMAGE_LIST_URL, tmp_dir / "totalfiles_jpg_512.csv")
        raw_goes_catalog = _download(GOES_CATALOG_URL, tmp_dir / "goes_flares_integrated.csv")

        image_index_path = tmp_dir / "image_index.csv"
        catalog_path = tmp_dir / "flare_catalog.csv"

        print("\nAdapting image index from filenames...")
        # The lab's totalfiles_jpg_512.csv isn't a pure image-path list -- it
        # also has a handful of unrelated labels_store/*.csv bookkeeping paths
        # mixed in (observed: 80 out of ~63.7k lines). build_image_index_from_filenames
        # intentionally raises ValueError rather than silently skipping garbage,
        # so filter to real .jpg paths here, at the call site, where it's visible.
        with open(raw_image_list, encoding="utf-8-sig") as f:
            all_paths = [line.strip() for line in f if line.strip()]
        image_paths = [p for p in all_paths if p.endswith(".jpg")]
        skipped = len(all_paths) - len(image_paths)
        if skipped:
            print(f"  skipped {skipped} non-.jpg entries out of {len(all_paths)} lines")

        image_index = build_image_index_from_filenames(image_paths, output_path=image_index_path)
        print(f"  {len(image_index)} images, timestamp range "
              f"{image_index['timestamp'].min()} -> {image_index['timestamp'].max()}")

        print("\nAdapting GOES catalog...")
        catalog = adapt_goes_catalog(raw_goes_catalog, output_path=catalog_path)
        print(f"  {len(catalog)} flares, classes seen: {sorted(catalog['class'].str[0].unique())}")
        print(f"  active_region non-null count: {catalog['active_region'].notna().sum()} / {len(catalog)}")

        print("\nRunning DatasetBuilder.build() on adapted data...")
        builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy(threshold="C"))
        result = builder.build(image_index_path, catalog_path)

        print(f"  built {len(result)} labeled rows")
        print(f"  label counts:\n{result['label'].value_counts()}")
        print("\nSample rows:")
        print(result.head())

        print("\nEnd-to-end chain completed successfully.")


if __name__ == "__main__":
    main()
