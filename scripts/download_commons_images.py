"""
Wikimedia Commons から学習候補画像を収集するスクリプト。

画像本体だけでなく、元ページ URL、作者、ライセンス情報も metadata.json に保存する。
収集後は必ず人手で内容とライセンスを確認してください。

使い方:
  python scripts/download_commons_images.py
  python scripts/download_commons_images.py --queries katana iaido kenjutsu --limit-per-query 30
  python scripts/download_commons_images.py --output dataset/raw_candidates/commons --image-width 1280
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path


API_URL = "https://commons.wikimedia.org/w/api.php"
DEFAULT_QUERIES = (
    "katana",
    "iaido",
    "kenjutsu",
    "japanese sword",
)
USER_AGENT = "form-sensei-dataset-tool/0.1"
MAX_RETRIES = 4


def request_json(params):
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_search_results(query: str, limit: int, image_width: int):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo|info",
        "inprop": "url",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": image_width,
    }
    data = request_json(params)
    pages = data.get("query", {}).get("pages", {})
    return sorted(pages.values(), key=lambda item: item.get("title", ""))


def sanitize_filename(value: str):
    safe = re.sub(r'[<>:"/\\\\|?*]+', "_", value)
    safe = safe.replace(" ", "_")
    return safe[:180]


def extract_metadata_field(extmetadata, key: str):
    raw = extmetadata.get(key, {}).get("value")
    if not raw:
        return None
    return html.unescape(re.sub(r"<[^>]+>", "", str(raw))).strip() or None


def build_record(page, query: str):
    imageinfo = (page.get("imageinfo") or [{}])[0]
    extmetadata = imageinfo.get("extmetadata") or {}

    image_url = imageinfo.get("thumburl") or imageinfo.get("url")
    if not image_url:
        return None

    title = page.get("title", "")
    original_name = title.removeprefix("File:")
    extension = Path(urllib.parse.urlparse(image_url).path).suffix or Path(original_name).suffix or ".jpg"
    stem = sanitize_filename(Path(original_name).stem or title or f"commons_{page.get('pageid', 'unknown')}")
    filename = f"{stem}{extension}"

    return {
        "query": query,
        "page_id": page.get("pageid"),
        "title": title,
        "filename": filename,
        "image_url": image_url,
        "source_page_url": page.get("fullurl"),
        "mime": imageinfo.get("mime"),
        "width": imageinfo.get("thumbwidth") or imageinfo.get("width"),
        "height": imageinfo.get("thumbheight") or imageinfo.get("height"),
        "license_short_name": extract_metadata_field(extmetadata, "LicenseShortName"),
        "license_url": extract_metadata_field(extmetadata, "LicenseUrl"),
        "artist": extract_metadata_field(extmetadata, "Artist"),
        "credit": extract_metadata_field(extmetadata, "Credit"),
        "usage_terms": extract_metadata_field(extmetadata, "UsageTerms"),
    }


def download_file(url: str, destination: Path):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                destination.write_bytes(response.read())
            return
        except HTTPError as exc:
            if exc.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after else 3.0 * (attempt + 1)
            time.sleep(wait_seconds)


def safe_console_text(value):
    return str(value).encode("cp932", errors="replace").decode("cp932")


def load_existing_metadata(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Wikimedia Commons から学習候補画像を収集")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=list(DEFAULT_QUERIES),
        help="検索クエリの一覧",
    )
    parser.add_argument(
        "--limit-per-query",
        type=int,
        default=25,
        help="クエリごとの取得件数",
    )
    parser.add_argument(
        "--output",
        default="dataset/raw_candidates/commons",
        help="保存先ディレクトリ",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1600,
        help="保存する画像の最大幅",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="ダウンロード間隔",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    seen_page_ids = set()
    records = load_existing_metadata(metadata_path)
    for record in records:
        if record.get("page_id") is not None:
            seen_page_ids.add(record["page_id"])

    for query in args.queries:
        print(f"search: {query}")
        try:
            pages = fetch_search_results(query, args.limit_per_query, args.image_width)
        except Exception as exc:
            print(f"  エラー: 検索失敗: {exc}")
            continue

        for page in pages:
            page_id = page.get("pageid")
            if page_id in seen_page_ids:
                continue

            record = build_record(page, query)
            if not record:
                continue

            destination = images_dir / record["filename"]
            suffix_counter = 2
            while destination.exists():
                destination = images_dir / f"{Path(record['filename']).stem}_{suffix_counter}{Path(record['filename']).suffix}"
                suffix_counter += 1
            record["filename"] = destination.name
            record["local_path"] = str(destination)

        try:
            download_file(record["image_url"], destination)
        except Exception as exc:
            print(f"  スキップ: {safe_console_text(record['title'])} のダウンロード失敗: {safe_console_text(exc)}")
            continue

            seen_page_ids.add(page_id)
            records.append(record)
            print(f"  saved: {destination.name}")
            time.sleep(args.sleep_seconds)

    metadata_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"完了: {len(records)} 件")
    print(f"画像: {images_dir}")
    print(f"metadata: {metadata_path}")
    print("次のステップ:")
    print("  1. 画像を人手で確認して、使える写真だけ残す")
    print("  2. train / val に振り分ける")
    print("  3. labelImg で katana_tip / katana_grip を付与する")

    if not records:
        sys.exit(1)


if __name__ == "__main__":
    main()
