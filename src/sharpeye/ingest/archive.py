"""Archive and dataset path resolution for batch QC."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from sharpeye.exceptions import ArchiveError

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
})

ARCHIVE_EXTENSIONS = frozenset({".zip"})

MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_FILES_PER_ARCHIVE = 10_000
MAX_MEMBER_BYTES = 50 * 1024 * 1024


def is_archive(path: Path) -> bool:
    return path.suffix.lower() in ARCHIVE_EXTENSIONS


def _scan_images(root: Path) -> list[Path]:
    if not root.exists():
        raise ArchiveError(f"Dataset path not found: {root}")

    if root.is_file():
        if is_archive(root):
            raise ArchiveError(
                "Archive paths must be opened with collect_images() context manager."
            )
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ArchiveError(f"Not an image file: {root}")
        return [root]

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ArchiveError(f"No images found under: {root}")
    return files


def _safe_extract_zip(archive: Path, dest: Path) -> None:
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ArchiveError(f"Archive too large (max {MAX_ARCHIVE_BYTES} bytes).")

    extracted = 0
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            extracted += 1
            if extracted > MAX_FILES_PER_ARCHIVE:
                raise ArchiveError(
                    f"Too many files in archive (max {MAX_FILES_PER_ARCHIVE})."
                )

            if info.file_size > MAX_MEMBER_BYTES:
                raise ArchiveError(
                    f"Archive member too large: {info.filename} "
                    f"(max {MAX_MEMBER_BYTES} bytes)."
                )

            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ArchiveError(f"Unsafe archive path: {info.filename}")

            target = (dest / member).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ArchiveError(f"Zip-slip blocked: {info.filename}")

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


@contextmanager
def collect_images(dataset: Path) -> Iterator[list[Path]]:
    """Yield image paths from a folder, single image, or .zip archive."""
    dataset = dataset.expanduser().resolve()

    if is_archive(dataset):
        with tempfile.TemporaryDirectory(prefix="sharpeye_") as tmp:
            root = Path(tmp)
            _safe_extract_zip(dataset, root)
            yield _scan_images(root)
        return

    yield _scan_images(dataset)