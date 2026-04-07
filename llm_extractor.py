"""
Academic paper entity and relation extraction with a fixed paper schema.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List

import anthropic
from openai import OpenAI

from config import (
    ANTHROPIC_API_KEY,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DEFAULT_LLM_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)


ALLOWED_ENTITY_TYPES = ("TASK", "METHOD", "DATASET", "METRIC", "RESULT")
ALLOWED_RELATION_TYPES = (
    "STUDIES",
    "PROPOSES",
    "USES_DATASET",
    "EVALUATED_BY",
    "APPLIED_TO",
    "TESTED_ON",
    "ACHIEVES",
)
RELATION_TARGET_TYPES = {
    "STUDIES": "TASK",
    "PROPOSES": "METHOD",
    "USES_DATASET": "DATASET",
    "EVALUATED_BY": "METRIC",
    "APPLIED_TO": "TASK",
    "TESTED_ON": "DATASET",
    "ACHIEVES": "RESULT",
}
ENTITY_LIMIT_PER_TYPE = 3
DEFAULT_DASHSCOPE_BASE_URLS = (
    "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL_PREFIXES = ("qwen", "qwq")


class LLMExtractor:
    """Extract paper-domain entities and relations from title and abstract."""

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL, provider: str = "openai"):
        self.model_name = model_name
        self.provider = provider.lower()
        self.logger = logging.getLogger(__name__)

        if self.provider in {"openai", "dashscope"}:
            api_key, base_urls = self._resolve_openai_compatible_config()
            if not api_key:
                if self.provider == "dashscope":
                    raise ValueError("DASHSCOPE_API_KEY not found in environment")
                raise ValueError("OPENAI_API_KEY not found in environment")

            self.openai_clients: List[tuple[str | None, OpenAI]] = []
            for base_url in base_urls:
                client_kwargs = {"api_key": api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url
                self.openai_clients.append((base_url, OpenAI(**client_kwargs)))
        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _resolve_openai_compatible_config(self) -> tuple[str | None, List[str | None]]:
        model_name = self.model_name.lower().strip()
        dashscope_model = model_name.startswith(QWEN_MODEL_PREFIXES)

        if self.provider == "dashscope" or dashscope_model:
            explicit_base_url = DASHSCOPE_BASE_URL or OPENAI_BASE_URL
            return (
                DASHSCOPE_API_KEY or OPENAI_API_KEY,
                [explicit_base_url] if explicit_base_url else list(DEFAULT_DASHSCOPE_BASE_URLS),
            )

        return OPENAI_API_KEY, [OPENAI_BASE_URL] if OPENAI_BASE_URL else [None]

    def _normalize_name(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip())

    def _lookup_key(self, value: str) -> str:
        return self._normalize_name(value).lower()

    def _extract_json_text(self, response: str) -> str:
        response = response.strip()
        match = re.search(r"\{.*\}", response, re.DOTALL)
        return match.group(0) if match else response

    def _create_extraction_prompt(self, paper_record: Dict[str, Any]) -> str:
        year = paper_record.get("year") or ""
        authors = ", ".join(paper_record.get("authors") or [])
        return f"""
You are an academic information extraction system.
Extract only the following entity types from the paper title and abstract: PAPER, TASK, METHOD, DATASET, METRIC, RESULT.
Return valid JSON only.
Do not invent information that is not explicitly supported by the text.

Constraints:
- At most 1 to 3 entities for each of TASK, METHOD, DATASET, METRIC, RESULT.
- Use the literal string "paper" as the source when a relation starts from the paper itself.
- If the evidence is weak or uncertain, return an empty array for that entity type instead of guessing.
- Keep entity names short and canonical.

Return this exact JSON shape:
{{
  "paper": {{
    "title": "{paper_record.get('title', '')}",
    "year": {json.dumps(paper_record.get('year'))}
  }},
  "entities": [
    {{"type": "TASK", "name": "..."}},
    {{"type": "METHOD", "name": "..."}},
    {{"type": "DATASET", "name": "..."}},
    {{"type": "METRIC", "name": "..."}},
    {{"type": "RESULT", "name": "..."}}
  ],
  "relations": [
    {{"source": "paper", "target": "...", "type": "STUDIES"}},
    {{"source": "paper", "target": "...", "type": "PROPOSES"}},
    {{"source": "paper", "target": "...", "type": "USES_DATASET"}},
    {{"source": "paper", "target": "...", "type": "EVALUATED_BY"}},
    {{"source": "...", "target": "...", "type": "APPLIED_TO"}},
    {{"source": "...", "target": "...", "type": "TESTED_ON"}},
    {{"source": "...", "target": "...", "type": "ACHIEVES"}}
  ]
}}

Paper metadata:
Title: {paper_record.get("title", "")}
Abstract: {paper_record.get("abstract", "")}
Year: {year}
Authors: {authors}
""".strip()

    def _call_openai(self, prompt: str) -> str:
        last_error: Exception | None = None
        for base_url, client in self.openai_clients:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You extract structured academic knowledge. "
                                "Always return valid JSON with no markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                )
                return response.choices[0].message.content or "{}"
            except Exception as exc:
                last_error = exc
                if len(self.openai_clients) > 1:
                    self.logger.warning(
                        "OpenAI-compatible request failed via %s: %s",
                        base_url or "default endpoint",
                        exc,
                    )

        if last_error is None:
            raise RuntimeError("No OpenAI-compatible client configured.")
        raise last_error

    def _call_anthropic(self, prompt: str) -> str:
        response = self.anthropic_client.messages.create(
            model=self.model_name,
            max_tokens=2000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _clean_entities(self, raw_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        counts = defaultdict(int)

        for raw_entity in raw_entities:
            if not isinstance(raw_entity, dict):
                continue
            entity_type = str(raw_entity.get("type", "")).upper().strip()
            name = self._normalize_name(str(raw_entity.get("name", "")))
            if entity_type not in ALLOWED_ENTITY_TYPES or not name:
                continue
            if counts[entity_type] >= ENTITY_LIMIT_PER_TYPE:
                continue

            key = (entity_type, self._lookup_key(name))
            if key in seen:
                continue

            entities.append(
                {
                    "type": entity_type,
                    "name": name,
                    "description": self._normalize_name(str(raw_entity.get("description", ""))),
                }
            )
            seen.add(key)
            counts[entity_type] += 1

        return entities

    def _clean_relations(
        self,
        raw_relations: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entity_by_name = {self._lookup_key(entity["name"]): entity for entity in entities}
        relations: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for raw_relation in raw_relations:
            if not isinstance(raw_relation, dict):
                continue

            relation_type = str(raw_relation.get("type", "")).upper().strip()
            source = self._normalize_name(str(raw_relation.get("source", "")))
            target = self._normalize_name(str(raw_relation.get("target", "")))
            if relation_type not in ALLOWED_RELATION_TYPES or not target:
                continue

            target_entity = entity_by_name.get(self._lookup_key(target))
            if target_entity is None or target_entity["type"] != RELATION_TARGET_TYPES[relation_type]:
                continue

            if source.lower() == "paper":
                canonical_source = "paper"
            else:
                source_entity = entity_by_name.get(self._lookup_key(source))
                if source_entity is None or source_entity["type"] != "METHOD":
                    continue
                if relation_type not in {"APPLIED_TO", "TESTED_ON", "ACHIEVES"}:
                    continue
                canonical_source = source_entity["name"]

            canonical_target = target_entity["name"]
            relation_key = (canonical_source.lower(), relation_type, canonical_target.lower())
            if relation_key in seen:
                continue

            relations.append(
                {
                    "source": canonical_source,
                    "target": canonical_target,
                    "type": relation_type,
                }
            )
            seen.add(relation_key)

        return relations

    def _parse_paper_response(self, response: str, paper_record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            payload = json.loads(self._extract_json_text(response))
        except json.JSONDecodeError as exc:
            self.logger.error("JSON parsing error: %s", exc)
            payload = {}

        raw_entities = payload.get("entities") or []
        raw_relations = payload.get("relations") or payload.get("relationships") or []
        entities = self._clean_entities(raw_entities)
        relations = self._clean_relations(raw_relations, entities)

        return {
            "paper": {
                "paper_id": paper_record.get("paper_id"),
                "title": paper_record.get("title"),
                "year": paper_record.get("year"),
                "authors": paper_record.get("authors") or [],
                "abstract": paper_record.get("abstract"),
                "text": paper_record.get("text"),
            },
            "entities": entities,
            "relations": relations,
        }

    def extract_paper_knowledge(self, paper_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the academic schema from one paper record."""
        prompt = self._create_extraction_prompt(paper_record)
        try:
            if self.provider in {"openai", "dashscope"}:
                response = self._call_openai(prompt)
            else:
                response = self._call_anthropic(prompt)
            return self._parse_paper_response(response, paper_record)
        except Exception as exc:
            self.logger.error("Paper extraction failed: %s", exc)
            return {
                "paper": {
                    "paper_id": paper_record.get("paper_id"),
                    "title": paper_record.get("title"),
                    "year": paper_record.get("year"),
                    "authors": paper_record.get("authors") or [],
                    "abstract": paper_record.get("abstract"),
                    "text": paper_record.get("text"),
                },
                "entities": [],
                "relations": [],
            }

    def extract_entities_and_relationships(self, text: str) -> Dict[str, Any]:
        """Compatibility wrapper for the previous generic API."""
        pseudo_paper = {
            "paper_id": "text-input",
            "title": "Untitled Paper",
            "abstract": text,
            "year": None,
            "authors": [],
            "text": text,
        }
        result = self.extract_paper_knowledge(pseudo_paper)
        return {
            "entities": result["entities"],
            "relationships": result["relations"],
            "paper": result["paper"],
        }

    def extract_entities_only(self, text: str) -> List[Dict[str, Any]]:
        return self.extract_entities_and_relationships(text)["entities"]

    def extract_relationships_only(self, text: str) -> List[Dict[str, Any]]:
        return self.extract_entities_and_relationships(text)["relationships"]

    def batch_extract(self, paper_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for index, paper_record in enumerate(paper_records, start=1):
            self.logger.info("Processing paper %s/%s", index, len(paper_records))
            results.append(self.extract_paper_knowledge(paper_record))
        return results


if __name__ == "__main__":
    print("LLMExtractor is configured for academic paper extraction.")
