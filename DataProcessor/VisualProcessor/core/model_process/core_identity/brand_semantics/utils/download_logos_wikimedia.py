#!/usr/bin/env python3
"""
Download brand logo images from Wikimedia Commons into `known_brands/`.

Why Wikimedia Commons:
- Stable API (no browser automation / no Google scraping).
- Supports thumbnails for SVG (returns PNG thumburl), which OpenCV can read.

Output:
  known_brands/<brand_path>/<N>.<ext>
Where <brand_path> can include grouping like `car/ferrari`.

Examples:
  # Download 15 candidate logo images for two brands (nested paths supported)
  python download_logos_wikimedia.py --brands "car/ferrari,car/lamborghini" --per-brand 15

  # Brands file (one per line, optional custom query):
  #   car/ferrari|Ferrari logo
  #   wear/nike|Nike logo
  python download_logos_wikimedia.py --brands-file brands.txt --per-brand 20 --thumb-width 768
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_KNOWN_ROOT = (
    "known_brands_auto"
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Wikimedia may rate-limit / ban arbitrary thumbnail widths. Keep to a conservative
# set of commonly supported sizes (and fall back to lower sizes on 429).
# If you need more, expand this list.
COMMONS_THUMB_WIDTH_STEPS = [
    120,
    160,
    200,
    240,
    320,
    400,
    500,
    640,
    800,
    1024,
    1280,
    1600,
    1920,
]


@dataclass(frozen=True)
class BrandSpec:
    brand_path: str  # e.g. "car/ferrari" or "nike"
    query: str  # e.g. "Ferrari logo"


def _split_csv(s: str) -> List[str]:
    parts = []
    for p in s.replace(";", ",").split(","):
        p = p.strip()
        if p:
            parts.append(p)
    return parts


def _sanitize_rel_path(p: str) -> str:
    # Keep nested paths like "car/ferrari", but prevent path traversal.
    p = p.strip().replace("\\", "/")
    p = re.sub(r"/+", "/", p)
    p = p.strip("/")
    if not p:
        raise ValueError("Empty brand path")
    if p.startswith(".") or "/.." in f"/{p}/" or p == "..":
        raise ValueError(f"Unsafe brand path: {p!r}")
    return p


def _brand_name_from_path(brand_path: str) -> str:
    return _sanitize_rel_path(brand_path).split("/")[-1]


def _default_query_for_brand(brand_path: str) -> str:
    # Use brand leaf; commons search works fine with ascii/lowercase too.
    leaf = _brand_name_from_path(brand_path)
    return f"{leaf} logo"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "download_logos_wikimedia",
        description="Download candidate brand logo images from Wikimedia Commons into known_brands/.",
    )
    ap.add_argument(
        "--known-root",
        default=DEFAULT_KNOWN_ROOT,
        help="Path to known_brands root (can be relative to repo).",
    )
    ap.add_argument(
        "--brands",
        default="",
        help="Comma-separated list of brand paths (supports nested like car/ferrari).",
    )
    ap.add_argument(
        "--brands-file",
        default="",
        help=(
            "Text file with one brand per line. Formats:\n"
            "  brand_path\n"
            "  brand_path|custom query\n"
            "Lines starting with # are ignored."
        ),
    )
    ap.add_argument("--per-brand", type=int, default=15, help="How many files to download per brand.")
    ap.add_argument(
        "--thumb-width",
        type=int,
        default=640,
        help="Thumbnail width in px (SVG will become PNG thumbnail).",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between API calls / downloads (seconds).",
    )
    ap.add_argument(
        "--min-bytes",
        type=int,
        default=5_000,
        help="Skip downloads smaller than this (often icons/blank).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download, only print planned files.",
    )
    return ap.parse_args()


def _read_brands_file(path: str) -> List[BrandSpec]:
    specs: List[BrandSpec] = []
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            brand_path, query = line.split("|", 1)
            brand_path = _sanitize_rel_path(brand_path)
            query = requests.utils.unquote(query.strip())
            if not query:
                query = _default_query_for_brand(brand_path)
        else:
            brand_path = _sanitize_rel_path(line)
            query = _default_query_for_brand(brand_path)
        specs.append(BrandSpec(brand_path=brand_path, query=query))
    return specs


def _build_specs(brands_csv: str, brands_file: str) -> List[BrandSpec]:
    specs: List[BrandSpec] = []
    if brands_csv:
        for b in _split_csv(brands_csv):
            bp = _sanitize_rel_path(b)
            specs.append(BrandSpec(brand_path=bp, query=_default_query_for_brand(bp)))
    if brands_file:
        specs.extend(_read_brands_file(brands_file))
    # Dedup by brand_path preserving order (last wins query)
    dedup: Dict[str, BrandSpec] = {}
    for s in specs:
        dedup[s.brand_path] = s
    return list(dedup.values())


def _get_next_index_for_dir(brand_dir: Path) -> int:
    if not brand_dir.is_dir():
        return 1
    max_id = 0
    for p in brand_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        try:
            n = int(stem)
        except ValueError:
            continue
        max_id = max(max_id, n)
    return max_id + 1


def _commons_search_files(session: requests.Session, query: str, limit: int) -> List[str]:
    """
    Returns list of titles in File namespace, e.g. 'File:Ferrari logo.svg'
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srnamespace": "6",  # File:
        "srsearch": query,
        "srlimit": str(limit),
    }
    r = session.get(COMMONS_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out: List[str] = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title")
        if isinstance(title, str) and title.startswith("File:"):
            out.append(title)
    return out


def _chunks(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])


def _commons_imageinfo(
    session: requests.Session, titles: Sequence[str], thumb_width: int
) -> List[Tuple[str, str, str]]:
    """
    For each title returns (title, url, mime) preferring thumburl if available.
    """
    results: List[Tuple[str, str, str]] = []
    if not titles:
        return results
    # MediaWiki allows multiple titles separated by |
    for batch in _chunks(list(titles), 25):
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": str(thumb_width),
            "titles": "|".join(batch),
        }
        r = session.get(COMMONS_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for _page_id, page in pages.items():
            title = page.get("title")
            ii = (page.get("imageinfo") or [{}])[0] or {}
            url = ii.get("thumburl") or ii.get("url") or ""
            mime = ii.get("mime") or ""
            if isinstance(title, str) and isinstance(url, str) and url:
                results.append((title, url, str(mime)))
    return results


def _ext_from_url(url: str) -> str:
    # Thumb URLs often end with .png even for original svg
    m = re.search(r"\.([a-zA-Z0-9]{2,5})(?:\?|$)", url)
    if not m:
        return ".jpg"
    ext = "." + m.group(1).lower()
    if ext == ".jpeg":
        return ".jpg"
    return ext


def _pick_thumb_width(requested: int) -> int:
    """
    Pick a supported thumbnail width step, using nearest <= requested (or min step).
    """
    requested = int(requested)
    if requested in COMMONS_THUMB_WIDTH_STEPS:
        return requested
    lower = [w for w in COMMONS_THUMB_WIDTH_STEPS if w <= requested]
    if lower:
        return max(lower)
    return min(COMMONS_THUMB_WIDTH_STEPS)


def _extract_thumb_width(url: str) -> Optional[int]:
    m = re.search(r"/(\d+)px-", url)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _replace_thumb_width(url: str, new_w: int) -> str:
    return re.sub(r"/\d+px-", f"/{int(new_w)}px-", url, count=1)


def _download_bytes(session: requests.Session, url: str, *, max_retries: int = 4, sleep_base: float = 2.0) -> bytes:
    """
    Download bytes with basic 429 handling:
    - respect Retry-After header when present
    - if response body mentions thumbnail steps, try smaller allowed widths
    """
    cur_url = url
    tried_urls: set[str] = set()
    for attempt in range(max_retries):
        tried_urls.add(cur_url)
        r = session.get(cur_url, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r.content

        # 429 handling
        retry_after = r.headers.get("Retry-After")
        wait_s: float = sleep_base * (attempt + 1)
        if retry_after:
            try:
                wait_s = max(wait_s, float(retry_after))
            except Exception:
                pass

        body = ""
        try:
            body = (r.text or "")[:500]
        except Exception:
            body = ""

        # If this is "thumbnail steps" policy, try smaller valid width by rewriting the URL.
        if "thumbnail steps" in body or "w.wiki/GHai" in body:
            w = _extract_thumb_width(cur_url)
            if w is not None:
                candidates = [x for x in COMMONS_THUMB_WIDTH_STEPS if x < w]
                candidates.sort(reverse=True)
                for alt_w in candidates:
                    alt_url = _replace_thumb_width(cur_url, alt_w)
                    if alt_url in tried_urls:
                        continue
                    cur_url = alt_url
                    # small pause before retry
                    time.sleep(min(wait_s, 10.0))
                    break
                else:
                    time.sleep(wait_s)
            else:
                time.sleep(wait_s)
        else:
            time.sleep(wait_s)

    # Final attempt (raise)
    r = session.get(cur_url, timeout=60)
    r.raise_for_status()
    return r.content


def main() -> int:
    args = parse_args()
    specs = _build_specs(args.brands, args.brands_file)
    if not specs:
        print("No brands provided. Use --brands or --brands-file.")
        return 2

    known_root = Path(args.known_root)
    known_root.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "TrendFlowML/brand_semantics downloader (Wikimedia Commons API)"})

    thumb_width = _pick_thumb_width(int(args.thumb_width))
    if thumb_width != int(args.thumb_width):
        print(
            f"[info] thumb width adjusted: requested={int(args.thumb_width)} -> using={thumb_width} "
            f"(to reduce 429 on non-standard thumbnail sizes)"
        )

    total_downloaded = 0
    total_skipped = 0

    for spec in specs:
        brand_rel = _sanitize_rel_path(spec.brand_path)
        brand_dir = known_root / brand_rel
        brand_dir.mkdir(parents=True, exist_ok=True)
        idx = _get_next_index_for_dir(brand_dir)

        print(f"\n=== {brand_rel} | query={spec.query!r} ===")
        titles = _commons_search_files(session, spec.query, limit=max(args.per_brand * 3, 30))
        time.sleep(args.sleep)
        infos = _commons_imageinfo(session, titles, thumb_width=thumb_width)
        time.sleep(args.sleep)

        downloaded_here = 0
        for title, url, mime in infos:
            if downloaded_here >= args.per_brand:
                break
            ext = _ext_from_url(url)
            if ext not in IMG_EXTS:
                # Most thumbs are png/jpg; skip other types
                total_skipped += 1
                continue

            out_path = brand_dir / f"{idx}{ext}"
            idx += 1

            if args.dry_run:
                print(f"[dry] {out_path} <= {title} ({mime})")
                downloaded_here += 1
                continue

            try:
                content = _download_bytes(session, url)
                if len(content) < int(args.min_bytes):
                    total_skipped += 1
                    continue
                out_path.write_bytes(content)
                print(f"[ok] {out_path.name} <= {title}")
                downloaded_here += 1
                total_downloaded += 1
            except Exception as e:
                print(f"[warn] download failed for {title}: {e}")
                total_skipped += 1
            time.sleep(args.sleep)

        print(f"Downloaded: {downloaded_here}/{args.per_brand} into {brand_dir}")

    print("\n=== Done ===")
    print(f"Downloaded files: {total_downloaded}")
    print(f"Skipped files: {total_skipped}")
    print(f"known_root: {known_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


