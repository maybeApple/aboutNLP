"""
Document processing module specialized for paper-record ingestion.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from docx import Document
import tiktoken

from config import SUPPORTED_FORMATS


PAPER_FILE_FORMATS = {".csv", ".jsonl"}


class DocumentProcessor:
    """Handles paper-record loading and single-paper text construction."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.encoding = None
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            print(f"Warning: failed to load tiktoken encoding, using word-count fallback: {exc}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or a whitespace fallback."""
        if self.encoding is None:
            return len(text.split())
        return len(self.encoding.encode(text))

    def clean_text(self, text: str) -> str:
        """Normalize whitespace while preserving common punctuation."""
        text = text.replace("\ufeff", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _canonical_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _field_value(self, record: Dict[str, Any], *candidates: str) -> Any:
        key_map = {self._canonical_key(key): key for key in record}
        for candidate in candidates:
            actual_key = key_map.get(self._canonical_key(candidate))
            if actual_key is None:
                continue
            value = record.get(actual_key)
            if value is not None:
                return value
        return None

    def _parse_authors(self, authors: Any) -> List[str]:
        if authors is None:
            return []
        if isinstance(authors, list):
            return [self.clean_text(str(item)) for item in authors if self.clean_text(str(item))]

        text = str(authors).strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [self.clean_text(str(item)) for item in parsed if self.clean_text(str(item))]
            except json.JSONDecodeError:
                pass

        if ";" in text or "|" in text:
            parts = re.split(r"[;|]", text)
        else:
            parts = [text]
        return [self.clean_text(part) for part in parts if self.clean_text(part)]

    def build_paper_text(
        self,
        title: str,
        abstract: str,
        year: str | int | None = None,
        authors: List[str] | None = None,
    ) -> str:
        """Build a single unified text block for one paper."""
        lines = [
            f"Title: {self.clean_text(title)}",
            f"Abstract: {self.clean_text(abstract)}",
        ]
        if year not in (None, ""):
            lines.append(f"Year: {year}")
        if authors:
            lines.append(f"Authors: {', '.join(authors)}")
        return "\n".join(lines)

    def _new_ingestion_summary(self) -> Dict[str, Any]:
        return {
            "papers": [],
            "processed_papers": 0,
            "skipped_papers": 0,
            "total_input_records": 0,
            "warnings": [],
        }

    def _record_warning(self, summary: Dict[str, Any], message: str) -> None:
        summary["warnings"].append(message)
        summary["skipped_papers"] += 1
        self.logger.warning(message)

    def process_paper_record(
        self,
        title: str,
        abstract: str,
        year: str | int | None = None,
        authors: List[str] | None = None,
        paper_id: str | None = None,
        source: str | None = None,
    ) -> Dict[str, Any]:
        """Convert one paper record into the normalized paper payload."""
        clean_title = self.clean_text(title)
        clean_abstract = self.clean_text(abstract)
        if not clean_title:
            raise ValueError("Paper title is required.")
        if not clean_abstract:
            raise ValueError(f"Abstract is required for paper: {clean_title}")

        author_list = self._parse_authors(authors)
        text = self.build_paper_text(clean_title, clean_abstract, year=year, authors=author_list)
        normalized_id = paper_id or hashlib.md5(clean_title.encode("utf-8")).hexdigest()[:12]

        return {
            "paper_id": normalized_id,
            "title": clean_title,
            "abstract": clean_abstract,
            "year": str(year).strip() if year not in (None, "") else None,
            "authors": author_list,
            "source": source,
            "text": text,
            "chunks": [
                {
                    "id": f"{normalized_id}-chunk-0",
                    "text": text,
                    "tokens": self.count_tokens(text),
                    "start_char": 0,
                    "end_char": len(text),
                }
            ],
        }

    def load_papers_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """Load paper records from a CSV file."""
        summary = self._new_ingestion_summary()
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"No header found in CSV file: {csv_path}")

            for row_number, row in enumerate(reader, start=2):
                summary["total_input_records"] += 1
                title = self._field_value(row, "title", "paper_title")
                abstract = self._field_value(row, "abstract", "summary")
                if not title or not abstract:
                    self._record_warning(
                        summary,
                        f"Skipped CSV row {row_number}: missing title or abstract.",
                    )
                    continue
                try:
                    paper = self.process_paper_record(
                        title=str(title),
                        abstract=str(abstract),
                        year=self._field_value(row, "year", "published_year"),
                        authors=self._field_value(row, "authors", "author"),
                        paper_id=self._field_value(row, "paper_id", "id", "arxiv_id"),
                        source=os.path.basename(csv_path),
                    )
                except Exception as exc:
                    self._record_warning(summary, f"Skipped CSV row {row_number}: {exc}")
                    continue

                summary["papers"].append(paper)
                summary["processed_papers"] += 1
        return summary

    def load_papers_from_jsonl(self, jsonl_path: str) -> Dict[str, Any]:
        """Load paper records from a JSONL file."""
        summary = self._new_ingestion_summary()
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                summary["total_input_records"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._record_warning(
                        summary,
                        f"Skipped JSONL line {line_number}: invalid JSON ({exc.msg}).",
                    )
                    continue

                title = self._field_value(record, "title", "paper_title")
                abstract = self._field_value(record, "abstract", "summary")
                if not title or not abstract:
                    self._record_warning(
                        summary,
                        f"Skipped JSONL line {line_number}: missing title or abstract.",
                    )
                    continue
                try:
                    paper = self.process_paper_record(
                        title=str(title),
                        abstract=str(abstract),
                        year=self._field_value(record, "year", "published_year"),
                        authors=self._field_value(record, "authors", "author"),
                        paper_id=self._field_value(record, "paper_id", "id", "arxiv_id"),
                        source=os.path.basename(jsonl_path),
                    )
                except Exception as exc:
                    self._record_warning(summary, f"Skipped JSONL line {line_number}: {exc}")
                    continue

                summary["papers"].append(paper)
                summary["processed_papers"] += 1
        return summary

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from .docx files."""
        try:
            doc = Document(file_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
            return "\n".join(paragraphs).strip()
        except Exception as exc:
            raise ValueError(f"Failed to extract text from {file_path}: {exc}") from exc

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from .txt files."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                return file.read().strip()

    def load_document(self, file_path: str) -> str:
        """Load and extract text from supported text document formats."""
        suffix = Path(file_path).suffix.lower()
        supported_formats = set(SUPPORTED_FORMATS) | PAPER_FILE_FORMATS
        if suffix not in supported_formats:
            raise ValueError(f"Unsupported file format: {suffix}")

        if suffix == ".docx":
            return self.extract_text_from_docx(file_path)
        if suffix == ".txt":
            return self.extract_text_from_txt(file_path)
        raise ValueError(f"Use CSV or JSONL loaders for paper-record file: {file_path}")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a source file into paper records."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = Path(file_path).suffix.lower()
        if suffix == ".csv":
            source_payload = self.load_papers_from_csv(file_path)
        elif suffix == ".jsonl":
            source_payload = self.load_papers_from_jsonl(file_path)
        else:
            text = self.load_document(file_path)
            if not text:
                raise ValueError("No text extracted from document")

            title = Path(file_path).stem.replace("_", " ").strip() or "Untitled Paper"
            paper = self.process_paper_record(
                title=title,
                abstract=self.clean_text(text),
                source=os.path.basename(file_path),
            )
            source_payload = self._new_ingestion_summary()
            source_payload["papers"].append(paper)
            source_payload["processed_papers"] = 1
            source_payload["total_input_records"] = 1

        papers = source_payload["papers"]

        total_tokens = sum(paper["chunks"][0]["tokens"] for paper in papers)
        total_chunks = sum(len(paper["chunks"]) for paper in papers)
        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "papers": papers,
            "total_papers": len(papers),
            "processed_papers": source_payload["processed_papers"],
            "skipped_papers": source_payload["skipped_papers"],
            "total_input_records": source_payload["total_input_records"],
            "warnings": source_payload["warnings"],
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_chunk_tokens": total_tokens / total_chunks if total_chunks else 0,
        }


if __name__ == "__main__":
    processor = DocumentProcessor()
    sample = processor.process_paper_record(
        title="BERT for Named Entity Recognition",
        abstract="We evaluate BERT on CoNLL-2003 and report strong F1.",
        year=2019,
        authors=["Alice Example", "Bob Example"],
    )
    print(json.dumps(sample, indent=2, ensure_ascii=False))
