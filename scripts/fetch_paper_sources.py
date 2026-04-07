"""
Fetch paper records from multiple public APIs and export them as CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OPENALEX_BASE_URL = "https://api.openalex.org/works"
ARXIV_BASE_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_OUTPUT = Path("data") / "papers_bootstrap_240.csv"
TOPIC_QUERIES = [
    "natural language processing",
    "text summarization",
    "question answering",
    "machine translation",
    "named entity recognition",
    "information retrieval",
    "dialogue systems",
    "vision language models",
    "code search",
]


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


def normalize_title_key(title: str) -> str:
    return "".join(ch.lower() for ch in title if ch.isalnum())


def request_json(url: str, timeout: int = 30) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def reconstruct_openalex_abstract(abstract_index: Dict[str, List[int]] | None) -> str:
    if not abstract_index:
        return ""
    last_position = max(position for positions in abstract_index.values() for position in positions)
    words = [""] * (last_position + 1)
    for word, positions in abstract_index.items():
        for position in positions:
            words[position] = word
    return normalize_text(" ".join(words))


def fetch_openalex_records(query: str, max_records: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    page = 1

    while len(records) < max_records and page <= 3:
        params = urllib.parse.urlencode(
            {
                "search": query,
                "per-page": min(100, max_records),
                "page": page,
            }
        )
        payload = request_json(f"{OPENALEX_BASE_URL}?{params}")
        for result in payload.get("results", []):
            title = normalize_text(result.get("display_name", ""))
            abstract = reconstruct_openalex_abstract(result.get("abstract_inverted_index"))
            if not title or not abstract:
                continue

            authors = [
                normalize_text(authorship.get("author", {}).get("display_name", ""))
                for authorship in result.get("authorships", [])
                if normalize_text(authorship.get("author", {}).get("display_name", ""))
            ]
            records.append(
                {
                    "paper_id": f"openalex:{result.get('id', '').rsplit('/', 1)[-1]}",
                    "title": title,
                    "abstract": abstract,
                    "year": result.get("publication_year") or "",
                    "authors": ";".join(authors),
                    "source_db": "OpenAlex",
                    "source_url": result.get("id", ""),
                    "query": query,
                }
            )
            if len(records) >= max_records:
                break
        page += 1
        time.sleep(0.2)

    return records


def fetch_arxiv_records(query: str, max_records: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    start = 0
    batch_size = min(50, max_records)

    while len(records) < max_records and start < max_records * 2:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": start,
                "max_results": batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        feed = request_text(f"{ARXIV_BASE_URL}?{params}")
        root = ET.fromstring(feed)
        entries = root.findall("atom:entry", ATOM_NAMESPACE)
        if not entries:
            break

        for entry in entries:
            title = normalize_text(entry.findtext("atom:title", default="", namespaces=ATOM_NAMESPACE))
            abstract = normalize_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NAMESPACE))
            if not title or not abstract:
                continue

            authors = [
                normalize_text(author.findtext("atom:name", default="", namespaces=ATOM_NAMESPACE))
                for author in entry.findall("atom:author", ATOM_NAMESPACE)
                if normalize_text(author.findtext("atom:name", default="", namespaces=ATOM_NAMESPACE))
            ]
            entry_id = normalize_text(entry.findtext("atom:id", default="", namespaces=ATOM_NAMESPACE))
            published = normalize_text(entry.findtext("atom:published", default="", namespaces=ATOM_NAMESPACE))
            records.append(
                {
                    "paper_id": f"arxiv:{entry_id.rsplit('/', 1)[-1]}",
                    "title": title,
                    "abstract": abstract,
                    "year": published[:4] if published else "",
                    "authors": ";".join(authors),
                    "source_db": "arXiv",
                    "source_url": entry_id,
                    "query": query,
                }
            )
            if len(records) >= max_records:
                break

        start += batch_size
        time.sleep(0.2)

    return records


def deduplicate_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_titles: set[str] = set()

    for record in records:
        title_key = normalize_title_key(record["title"])
        if not title_key or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        deduped.append(record)

    return deduped


def build_dataset(target_count: int) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    per_query_openalex = max(20, target_count // 8)
    per_query_arxiv = max(12, target_count // 12)

    for query in TOPIC_QUERIES:
        collected.extend(fetch_openalex_records(query, per_query_openalex))
        collected.extend(fetch_arxiv_records(query, per_query_arxiv))
        deduped = deduplicate_records(collected)
        if len(deduped) >= target_count:
            return deduped[:target_count]

    return deduplicate_records(collected)[:target_count]


def write_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id",
        "title",
        "abstract",
        "year",
        "authors",
        "source_db",
        "source_url",
        "query",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 200+ paper records from public sources.")
    parser.add_argument("--count", type=int, default=240, help="Number of papers to export")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    records = build_dataset(args.count)
    if len(records) < args.count:
        raise RuntimeError(f"Only collected {len(records)} unique papers, below requested {args.count}.")

    output_path = Path(args.output)
    write_csv(records, output_path)

    source_counts: Dict[str, int] = {}
    for record in records:
        source_counts[record["source_db"]] = source_counts.get(record["source_db"], 0) + 1

    print(f"Exported {len(records)} papers to {output_path}")
    print(f"Source counts: {source_counts}")


if __name__ == "__main__":
    main()
