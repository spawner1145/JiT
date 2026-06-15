#!/usr/bin/env python3
"""Convert ImageNet parquet shards to torchvision ImageFolder layout.

The ModelScope/HuggingFace ImageNet parquet files store samples as
{"image": ..., "label": int}. JiT currently reads
torchvision.datasets.ImageFolder(<data_path>/train), so this script expands
the parquet shards into:

    output_dir/train/n01440764/*.jpg
    output_dir/train/n01443537/*.jpg
    ...

When the parquet directory contains files from multiple splits
(e.g. train-00000-of-00142.parquet, validation-00000-of-00014.parquet,
test-00000-of-00014.parquet), use --auto_split to automatically detect the
split name from each file's name prefix and write to the corresponding
output subdirectory.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import shutil
import sys
import re
from pathlib import Path
from typing import Any


VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}

PIL_FORMAT_TO_EXTENSION = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "PPM": ".ppm",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ModelScope/HuggingFace ImageNet parquet shards to ImageFolder."
    )
    parser.add_argument(
        "--parquet_dir",
        type=Path,
        required=True,
        help="Directory containing parquet shards.",
    )
    parser.add_argument(
        "--classes_py",
        type=Path,
        required=True,
        help="Path to classes.py defining IMAGENET2012_CLASSES.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="ImageNet root to create. The script writes <output_dir>/<output_split>/<synset>/...",
    )
    parser.add_argument(
        "--output_split",
        default="train",
        help="Output split directory name under output_dir.",
    )
    parser.add_argument(
        "--auto_split",
        action="store_true",
        help=(
            "Automatically detect the split name from each parquet file's "
            "name prefix (e.g. 'train-...', 'validation-...', 'test-...') "
            "and write to the corresponding output subdirectory. "
            "When set, --output_split is ignored."
        ),
    )
    parser.add_argument(
        "--split_prefixes",
        default="train,validation,val,test",
        help=(
            "Comma-separated list of known split prefixes to recognize in "
            "parquet filenames when --auto_split is used. "
            "Default: 'train,validation,val,test'."
        ),
    )
    parser.add_argument(
        "--parquet_glob",
        default="*.parquet",
        help="Glob used to select parquet files. Use e.g. 'train*.parquet' if all splits are in one folder.",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only search parquet_dir itself instead of searching recursively.",
    )
    parser.add_argument("--image_column", default="image", help="Image column name in parquet.")
    parser.add_argument("--label_column", default="label", help="Label column name in parquet.")
    parser.add_argument("--batch_size", type=int, default=512, help="Rows to read from parquet at a time.")
    parser.add_argument(
        "--class_order",
        choices=("sorted", "file"),
        default="sorted",
        help="Use sorted synset ids for label->class mapping, or preserve classes.py file order.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images whose deterministic output filename already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir/output_split before converting.",
    )
    parser.add_argument(
        "--skip_bad_images",
        action="store_true",
        help="Skip samples whose image bytes cannot be read or written.",
    )
    parser.add_argument(
        "--skip_invalid_labels",
        action="store_true",
        help="Skip labels outside [0, num_classes). Useful only when intentionally filtering mixed shards.",
    )
    parser.add_argument(
        "--reencode_jpeg",
        action="store_true",
        help="Decode every image and save RGB JPEG instead of preserving original encoded bytes.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality used with --reencode_jpeg.")
    parser.add_argument(
        "--verify_images",
        action="store_true",
        help="Ask PIL to verify each image before writing or copying. Slower, but useful for a first pass.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most this many valid samples. Useful for a smoke test.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10000,
        help="Print progress every N written or existing images.",
    )
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when set.")
    return args


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _string_literal(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise ValueError("Class keys in IMAGENET2012_CLASSES must be string literals.")


def _extract_ordered_keys(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Call) and _call_name(node.func) == "OrderedDict":
        if len(node.args) != 1:
            raise ValueError("Expected OrderedDict to receive exactly one positional argument.")
        return _extract_ordered_keys(node.args[0])

    if isinstance(node, ast.Dict):
        return [_string_literal(key) for key in node.keys]

    literal = ast.literal_eval(node)
    if isinstance(literal, dict):
        return [str(key) for key in literal.keys()]
    if isinstance(literal, (list, tuple)):
        return [str(item[0]) for item in literal]
    raise ValueError("Could not parse IMAGENET2012_CLASSES as a mapping.")


def load_synsets(classes_py: Path, class_order: str) -> list[str]:
    tree = ast.parse(classes_py.read_text(encoding="utf-8"), filename=str(classes_py))
    value_node: ast.AST | None = None

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IMAGENET2012_CLASSES":
                    value_node = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "IMAGENET2012_CLASSES":
                value_node = node.value
        if value_node is not None:
            break

    if value_node is None:
        raise ValueError(f"{classes_py} does not define IMAGENET2012_CLASSES.")

    synsets = _extract_ordered_keys(value_node)
    if len(set(synsets)) != len(synsets):
        raise ValueError("IMAGENET2012_CLASSES contains duplicate synset ids.")
    if len(synsets) != 1000:
        raise ValueError(f"Expected 1000 ImageNet classes, found {len(synsets)}.")

    if class_order == "sorted":
        synsets = sorted(synsets)
    return synsets


def discover_parquet_files(parquet_dir: Path, parquet_glob: str, recursive: bool) -> list[Path]:
    if recursive:
        files = sorted(path for path in parquet_dir.rglob(parquet_glob) if path.is_file())
    else:
        files = sorted(path for path in parquet_dir.glob(parquet_glob) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files matched {parquet_glob!r} under {parquet_dir}.")
    return files


def has_any_file(path: Path) -> bool:
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def prepare_output(split_dir: Path, synsets: list[str], overwrite: bool, resume: bool) -> None:
    if overwrite and split_dir.exists():
        shutil.rmtree(split_dir)
    if has_any_file(split_dir) and not resume:
        raise FileExistsError(
            f"{split_dir} already contains files. Use --resume to skip existing outputs "
            "or --overwrite to rebuild it."
        )
    for synset in synsets:
        (split_dir / synset).mkdir(parents=True, exist_ok=True)


def normalized_extension(ext: str | None) -> str | None:
    if not ext:
        return None
    ext = ext.lower()
    if ext in VALID_IMAGE_EXTENSIONS:
        return ".jpg" if ext == ".jpeg" else ext
    return None


def detect_extension_from_bytes(raw: bytes) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        return PIL_FORMAT_TO_EXTENSION.get((image.format or "").upper(), ".jpg")


def verify_image_bytes(raw: bytes) -> None:
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        image.verify()


def verify_image_path(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()


def write_bytes_atomic(raw: bytes, destination: Path) -> None:
    tmp = destination.with_name(destination.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(raw)
    os.replace(tmp, destination)


def copy_file_atomic(source: Path, destination: Path) -> None:
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)


def save_pil_as_jpeg_atomic(image: Any, destination: Path, quality: int) -> None:
    tmp = destination.with_name(destination.name + ".tmp")
    image.convert("RGB").save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, destination)


def resolve_image_path(path_value: str, parquet_file: Path, parquet_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates = [
        parquet_file.parent / path,
        parquet_dir / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def image_value_to_payload(image_value: Any, parquet_file: Path, parquet_dir: Path) -> tuple[str, Any, str | None]:
    if isinstance(image_value, dict):
        raw = image_value.get("bytes")
        path_value = image_value.get("path")
        if raw is not None:
            return "bytes", bytes(raw), str(path_value) if path_value else None
        if path_value:
            return "path", resolve_image_path(str(path_value), parquet_file, parquet_dir), str(path_value)
        raise ValueError("Image dict has neither 'bytes' nor 'path'.")

    if isinstance(image_value, (bytes, bytearray, memoryview)):
        return "bytes", bytes(image_value), None

    if isinstance(image_value, str):
        return "path", resolve_image_path(image_value, parquet_file, parquet_dir), image_value

    if hasattr(image_value, "save"):
        return "pil", image_value, None

    raise TypeError(f"Unsupported image value type: {type(image_value)!r}")


def open_payload_as_pil(kind: str, payload: Any) -> Any:
    from PIL import Image

    if kind == "bytes":
        image = Image.open(io.BytesIO(payload))
    elif kind == "path":
        image = Image.open(payload)
    elif kind == "pil":
        image = payload
    else:
        raise TypeError(f"Unsupported payload kind: {kind}")
    image.load()
    return image


def output_exists(base_path: Path) -> bool:
    return any(base_path.with_suffix(ext).exists() for ext in VALID_IMAGE_EXTENSIONS)


def save_image(
    image_value: Any,
    destination_base: Path,
    parquet_file: Path,
    parquet_dir: Path,
    args: argparse.Namespace,
) -> str:
    if args.resume and output_exists(destination_base):
        return "existing"

    kind, payload, path_hint = image_value_to_payload(image_value, parquet_file, parquet_dir)

    if args.reencode_jpeg:
        destination = destination_base.with_suffix(".jpg")
        if args.resume and destination.exists():
            return "existing"
        image = open_payload_as_pil(kind, payload)
        save_pil_as_jpeg_atomic(image, destination, args.jpeg_quality)
        return "written"

    if kind == "bytes":
        raw = payload
        ext = normalized_extension(Path(path_hint).suffix if path_hint else None)
        if ext is None:
            ext = detect_extension_from_bytes(raw)
        destination = destination_base.with_suffix(ext)
        if args.resume and destination.exists():
            return "existing"
        if args.verify_images:
            verify_image_bytes(raw)
        write_bytes_atomic(raw, destination)
        return "written"

    if kind == "path":
        source = Path(payload)
        if not source.exists():
            raise FileNotFoundError(f"Image path referenced by parquet does not exist: {source}")
        ext = normalized_extension(source.suffix)
        if ext is None:
            with source.open("rb") as handle:
                ext = detect_extension_from_bytes(handle.read())
        destination = destination_base.with_suffix(ext)
        if args.resume and destination.exists():
            return "existing"
        if args.verify_images:
            verify_image_path(source)
        copy_file_atomic(source, destination)
        return "written"

    destination = destination_base.with_suffix(".jpg")
    if args.resume and destination.exists():
        return "existing"
    image = open_payload_as_pil(kind, payload)
    save_pil_as_jpeg_atomic(image, destination, args.jpeg_quality)
    return "written"


def detect_split_from_filename(parquet_file: Path, known_prefixes: list[str]) -> str | None:
    """Try to extract a split name from a parquet file's name.

    Recognized patterns:
      - <split>-<shard_info>.parquet  →  <split>
        e.g. "train-00000-of-00142.parquet" → "train"
        e.g. "validation-00000-of-00014.parquet" → "validation"
    """
    stem = parquet_file.stem  # filename without .parquet
    # Match known prefix followed by a separator like '-' or '_'
    for prefix in known_prefixes:
        # e.g. "train-00000-..." matches prefix "train"
        pattern = rf"^{re.escape(prefix)}[-_]"
        if re.match(pattern, stem):
            return prefix
    # Also try the full stem if it is exactly a known prefix
    if stem in known_prefixes:
        return stem
    return None


def convert(args: argparse.Namespace) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install pyarrow to read parquet files: pip install pyarrow") from exc

    parquet_dir = args.parquet_dir.resolve()
    output_dir = args.output_dir.resolve()
    classes_py = args.classes_py.resolve()

    synsets = load_synsets(classes_py, args.class_order)
    parquet_files = discover_parquet_files(parquet_dir, args.parquet_glob, not args.no_recursive)

    auto_split = args.auto_split
    known_prefixes = [p.strip() for p in args.split_prefixes.split(",") if p.strip()]

    # When auto_split is enabled, group parquet files by detected split name.
    if auto_split:
        split_groups: dict[str, list[Path]] = {}
        unassigned: list[Path] = []
        for pf in parquet_files:
            split_name = detect_split_from_filename(pf, known_prefixes)
            if split_name is not None:
                split_groups.setdefault(split_name, []).append(pf)
            else:
                unassigned.append(pf)
        if unassigned:
            names = ", ".join(str(p.name) for p in unassigned)
            raise ValueError(
                f"--auto_split could not detect split from filenames: {names}. "
                "Either rename files to follow '<split>-<shard>.parquet' convention, "
                "or use --parquet_glob and --output_split manually."
            )
        print(f"Auto-detected splits: {', '.join(f'{k} ({len(v)} files)' for k, v in split_groups.items())}")
    else:
        # Single-split mode: all files go to args.output_split
        split_groups = {args.output_split: parquet_files}

    total_written = 0
    total_existing = 0
    total_skipped_invalid = 0
    total_skipped_bad = 0

    for split_name, split_parquet_files in split_groups.items():
        split_dir = output_dir / split_name
        prepare_output(split_dir, synsets, args.overwrite, args.resume)

        print(f"\n{'=' * 60}")
        print(f"Split: {split_name}  ({len(split_parquet_files)} parquet files)")
        print(f"Output: {split_dir}")
        print(f"{'=' * 60}")
        print(f"First parquet files:")
        for path in split_parquet_files[:5]:
            print(f"  {path}")
        if len(split_parquet_files) > 5:
            print("  ...")

        written = 0
        existing = 0
        skipped_invalid = 0
        skipped_bad = 0
        seen_valid = 0

        for file_index, parquet_file in enumerate(split_parquet_files):
            print(f"[{file_index + 1}/{len(split_parquet_files)}] {parquet_file}")
            parquet = pq.ParquetFile(parquet_file)
            schema_names = set(parquet.schema_arrow.names)
            missing = [name for name in (args.image_column, args.label_column) if name not in schema_names]
            if missing:
                raise KeyError(f"{parquet_file} is missing required columns: {missing}")

            row_offset = 0
            for batch in parquet.iter_batches(
                batch_size=args.batch_size,
                columns=[args.image_column, args.label_column],
            ):
                image_column = batch.column(batch.schema.get_field_index(args.image_column)).to_pylist()
                label_column = batch.column(batch.schema.get_field_index(args.label_column)).to_pylist()

                for offset, (image_value, label_value) in enumerate(zip(image_column, label_column)):
                    row_index = row_offset + offset
                    label = int(label_value)
                    if label < 0 or label >= len(synsets):
                        if args.skip_invalid_labels:
                            skipped_invalid += 1
                            continue
                        raise ValueError(
                            f"Invalid label {label} in {parquet_file} row {row_index}. "
                            "If this is a test split with label=-1, pass only train shards "
                            "or add --skip_invalid_labels intentionally."
                        )

                    synset = synsets[label]
                    destination_base = split_dir / synset / f"{file_index:05d}-{row_index:08d}"
                    try:
                        result = save_image(image_value, destination_base, parquet_file, parquet_dir, args)
                    except Exception:
                        if not args.skip_bad_images:
                            raise
                        skipped_bad += 1
                        continue

                    if result == "existing":
                        existing += 1
                    else:
                        written += 1
                    seen_valid += 1

                    if args.print_every > 0 and seen_valid % args.print_every == 0:
                        print(
                            f"  valid={seen_valid} written={written} existing={existing} "
                            f"skipped_invalid={skipped_invalid} skipped_bad={skipped_bad}"
                        )

                    if args.limit is not None and seen_valid >= args.limit:
                        print("Reached --limit; stopping early.")
                        print_summary(written, existing, skipped_invalid, skipped_bad, split_dir)
                        total_written += written
                        total_existing += existing
                        total_skipped_invalid += skipped_invalid
                        total_skipped_bad += skipped_bad
                        return

                row_offset += batch.num_rows

        print_summary(written, existing, skipped_invalid, skipped_bad, split_dir)
        total_written += written
        total_existing += existing
        total_skipped_invalid += skipped_invalid
        total_skipped_bad += skipped_bad

    if auto_split and len(split_groups) > 1:
        print(f"\n{'=' * 60}")
        print("Overall summary across all splits:")
        print(f"  total_written: {total_written}")
        print(f"  total_existing: {total_existing}")
        print(f"  total_skipped_invalid_labels: {total_skipped_invalid}")
        print(f"  total_skipped_bad_images: {total_skipped_bad}")
        print(f"  splits: {', '.join(split_groups.keys())}")


def print_summary(written: int, existing: int, skipped_invalid: int, skipped_bad: int, split_dir: Path) -> None:
    print("Done.")
    print(f"  output: {split_dir}")
    print(f"  written: {written}")
    print(f"  existing: {existing}")
    print(f"  skipped_invalid_labels: {skipped_invalid}")
    print(f"  skipped_bad_images: {skipped_bad}")


def main() -> int:
    args = parse_args()
    try:
        convert(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
