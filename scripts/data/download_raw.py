#!/usr/bin/env python3
"""Download the raw corpora used by the authorization/translation project.

The script intentionally uses only Python's standard library. Downloads are
written to ``*.part`` first, support HTTP range-resume when the server allows
it, and are opened as ZIP files before extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


DGT_BASE_URL = "https://wt-public.emm4u.eu/Resources/DGT-Acquis-2012"
DGT_FILES = {
    "a1": "data.a1.txt.zip",
    "en": "data.en.txt.zip",
    "fr": "data.fr.txt.zip",
    "de": "data.de.txt.zip",
}
NOTICEBENCH_URL = (
    "https://github.com/liamjxu/NoticeBench/archive/refs/heads/main.zip"
)
USER_AGENT = "authorization-translation-research/0.1"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def download_file(url: str, destination: Path, retries: int = 3) -> Path:
    """Download URL atomically, resuming a partial file where possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[skip] archive exists: {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = existing > 0 and response.status == 206
                mode = "ab" if resumed else "wb"
                if existing and not resumed:
                    existing = 0
                expected = response.headers.get("Content-Length")
                total = existing + int(expected) if expected else None
                downloaded = existing

                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = 100.0 * downloaded / total
                            msg = (
                                f"\r[download] {destination.name}: "
                                f"{_human_bytes(downloaded)} / "
                                f"{_human_bytes(total)} ({percent:5.1f}%)"
                            )
                        else:
                            msg = (
                                f"\r[download] {destination.name}: "
                                f"{_human_bytes(downloaded)}"
                            )
                        print(msg, end="", flush=True)
            print()
            partial.replace(destination)
            return destination
        except (OSError, urllib.error.URLError) as error:
            print(f"\n[retry {attempt}/{retries}] {url}: {error}", file=sys.stderr)
            if attempt == retries:
                raise

    raise AssertionError("unreachable")


def validate_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                raise ValueError(f"ZIP archive is empty: {path}")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Corrupt ZIP member {bad_member!r} in {path}")
    except zipfile.BadZipFile as error:
        raise ValueError(f"Not a valid ZIP archive: {path}") from error


def _safe_member_path(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return member


def extract_single_text(archive_path: Path, destination: Path) -> None:
    """Extract the only non-directory member and give it a stable name."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[skip] extracted file exists: {destination}")
        return
    with zipfile.ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(
                f"Expected one file in {archive_path}, found {len(members)}"
            )
        _safe_member_path(members[0].filename)
        temporary = destination.with_suffix(destination.suffix + ".part")
        with archive.open(members[0]) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, CHUNK_SIZE)
        temporary.replace(destination)


def extract_noticebench(archive_path: Path, destination: Path) -> None:
    """Extract the GitHub archive while removing its top-level directory."""
    if destination.exists():
        print(f"[skip] extracted directory exists: {destination}")
        return
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(
            f"Incomplete extraction exists at {temporary}; remove it and retry"
        )
    temporary.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member = _safe_member_path(info.filename)
                if len(member.parts) < 2:
                    continue
                relative = Path(*member.parts[1:])
                target = temporary / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, CHUNK_SIZE)
        temporary.replace(destination)
    except Exception:
        # Keep the partial directory for diagnosis; never delete user data.
        raise


def write_manifest(data_root: Path, entries: Iterable[dict]) -> None:
    manifest_path = data_root / "raw" / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous: list[dict] = []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "downloads", []
        )

    by_url = {entry["url"]: entry for entry in previous}
    for entry in entries:
        by_url[entry["url"]] = entry
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloads": sorted(by_url.values(), key=lambda row: row["url"]),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def download_dgt(data_root: Path, languages: list[str], keep_archives: bool) -> None:
    requested = ["a1", *languages]
    archive_dir = data_root / "_archives" / "dgt_acquis"
    raw_dir = data_root / "raw" / "dgt_acquis"
    manifest_entries = []

    for language in requested:
        filename = DGT_FILES[language]
        url = f"{DGT_BASE_URL}/{filename}"
        archive_path = archive_dir / filename
        output_path = raw_dir / f"{language}.txt"
        if not output_path.exists():
            download_file(url, archive_path)
            validate_zip(archive_path)
            extract_single_text(archive_path, output_path)
        else:
            print(f"[skip] DGT {language}: {output_path}")

        entry = {
            "dataset": "DGT-Acquis",
            "component": language,
            "url": url,
            "output": str(output_path.relative_to(data_root)),
        }
        if archive_path.exists():
            entry.update(
                archive_bytes=archive_path.stat().st_size,
                archive_sha256=sha256_file(archive_path),
            )
        manifest_entries.append(entry)

        if archive_path.exists() and not keep_archives:
            archive_path.unlink()

    write_manifest(data_root, manifest_entries)


def download_noticebench(data_root: Path, keep_archives: bool) -> None:
    archive_dir = data_root / "_archives" / "noticebench"
    archive_path = archive_dir / "NoticeBench-main.zip"
    output_dir = data_root / "raw" / "noticebench"

    if not output_dir.exists():
        download_file(NOTICEBENCH_URL, archive_path)
        validate_zip(archive_path)
        extract_noticebench(archive_path, output_dir)
    else:
        print(f"[skip] NoticeBench: {output_dir}")

    entry = {
        "dataset": "NoticeBench",
        "component": "repository-main",
        "url": NOTICEBENCH_URL,
        "output": str(output_dir.relative_to(data_root)),
    }
    if archive_path.exists():
        entry.update(
            archive_bytes=archive_path.stat().st_size,
            archive_sha256=sha256_file(archive_path),
        )
    write_manifest(data_root, [entry])

    if archive_path.exists() and not keep_archives:
        archive_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("dgt", "noticebench", "all"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Project data directory (default: data)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=sorted(key for key in DGT_FILES if key != "a1"),
        default=["en", "fr", "de"],
        help="DGT language columns to fetch (default: en fr de)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep ZIP archives after successful extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if args.source in ("dgt", "all"):
        download_dgt(data_root, args.languages, args.keep_archives)
    if args.source in ("noticebench", "all"):
        download_noticebench(data_root, args.keep_archives)
    print(f"[done] raw data root: {data_root / 'raw'}")


if __name__ == "__main__":
    main()

