"""Loading, validating, and bounded retrieval of real evaluation cases."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import EvalCase


MANIFEST_SCHEMA_VERSION = 1
MODE_CASE_COUNTS = {"smoke": 20, "improvement": 30, "validation": 30}


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    checked: int


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"evaluation data is missing: {path}")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value if isinstance(value, list) else value.get("cases", [])
    if not isinstance(records, list):
        raise ValueError(f"expected a list of cases in {path}")
    return records


def load_cases(path: str | Path, *, source: str | None = None) -> list[EvalCase]:
    cases = [EvalCase.from_dict(value) for value in _read_records(Path(path))]
    if source is not None:
        cases = [case for case in cases if case.source == source]
    if not cases:
        raise ValueError(f"no real evaluation cases found in {path}")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id in evaluation data")
    return cases


def validate_project_cases(cases: Iterable[EvalCase], root: str | Path = ".") -> ValidationReport:
    errors: list[str] = []
    checked = 0
    root_path = Path(root)
    for case in cases:
        if case.source != "project":
            continue
        checked += 1
        for span in case.source_spans:
            path = root_path / span.path
            if not path.exists():
                errors.append(f"{case.case_id}: source file does not exist: {span.path}")
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            if span.end_line > len(lines):
                errors.append(f"{case.case_id}: source span exceeds {span.path}")
                continue
            actual = "\n".join(lines[span.start_line - 1 : span.end_line])
            if actual != span.quote:
                errors.append(f"{case.case_id}: source quote changed at {span.path}:{span.start_line}")
    return ValidationReport(not errors, tuple(errors), checked)


def canonical_row_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _get_json(url: str, timeout: float = 30.0) -> Mapping[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "prompt-engineering-evaluation/1.0"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Hugging Face API unavailable: {url}: {last_error}") from last_error


def fetch_huggingface_cases(manifest: Mapping[str, Any], split: str) -> list[EvalCase]:
    """Fetch only the rows named by a manifest and verify immutable hashes.

    This uses the Hugging Face datasets-server first-rows endpoint, rather than
    downloading a dataset snapshot.  A missing network or changed row is a
    hard error; mock/fallback records are intentionally not supported.
    """
    dataset = manifest["huggingface"]
    metadata = _get_json(dataset["metadata_api"])
    verify_huggingface_metadata(manifest, metadata)
    split_info = dataset["splits"][split]
    selection = split_info["selection"]
    expected_count = int(split_info["count"])
    if len(selection) != expected_count:
        raise ValueError(f"manifest selection for {split} has {len(selection)} rows, expected {expected_count}")
    source_split = split_info["source_split"]
    params = urllib.parse.urlencode({"dataset": dataset["id"], "config": dataset["config"], "split": source_split})
    api_url = f"{dataset['rows_api']}?{params}"
    payload = _get_json(api_url)
    actual = {int(row["row_idx"]): row["row"] for row in payload.get("rows", [])}
    cases: list[EvalCase] = []
    for entry in selection:
        index = int(entry["row_idx"])
        row = actual.get(index)
        if row is None:
            raise RuntimeError(f"Hugging Face API did not return required row {split}:{index}")
        if str(row.get("id")) != str(entry["row_id"]):
            raise RuntimeError(f"Hugging Face row ID changed for {split}:{index}; expected {entry['row_id']}, got {row.get('id')}")
        digest = canonical_row_hash(row)
        if digest != entry["row_hash"]:
            raise RuntimeError(f"Hugging Face row changed for {split}:{index}; expected {entry['row_hash']}, got {digest}")
        answers = row.get("answers", {})
        texts = answers.get("text", [])
        starts = answers.get("answer_start", [])
        if not texts:
            raise ValueError(f"Hugging Face row has no answer: {split}:{index}")
        case_id = f"hf-squad-{split}-{row['id']}"
        cases.append(EvalCase(
            case_id=case_id,
            question=row["question"],
            expected_answer=texts[0],
            relevant_document_ids=(f"hf-squad-context-{row['id']}",),
            source="huggingface",
            dataset_id=dataset["id"], split=source_split, row_id=row["id"], row_hash=digest,
            answer_spans=tuple({"start": int(start), "end": int(start) + len(text)} for start, text in zip(starts, texts)),
        ))
    return cases


def validate_split_disjointness(manifest: Mapping[str, Any]) -> None:
    """Fail if bounded improvement and validation selections overlap."""
    splits = manifest["huggingface"]["splits"]
    left = {entry["row_id"] for entry in splits["improvement"]["selection"]}
    right = {entry["row_id"] for entry in splits["validation"]["selection"]}
    overlap = left & right
    if overlap:
        raise ValueError(f"improvement and validation rows overlap: {sorted(overlap)}")


def verify_huggingface_metadata(manifest: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    """Check the run-time API response against recorded provenance."""
    expected = manifest["huggingface"]
    if metadata.get("sha") != expected["revision"]:
        raise RuntimeError(f"Hugging Face revision changed: expected {expected['revision']}, got {metadata.get('sha')}")
    card = metadata.get("cardData") or {}
    if card.get("license") != expected["license"]:
        raise RuntimeError(f"Hugging Face license changed: expected {expected['license']}, got {card.get('license')}")
    languages = card.get("language") or []
    if sorted(languages) != sorted(expected["language"]):
        raise RuntimeError(f"Hugging Face language metadata changed: expected {expected['language']}, got {languages}")


def load_manifest(path: str | Path) -> Mapping[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"evaluation manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("evaluation manifest must be a JSON object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation manifest schema")
    project = manifest.get("project_source")
    if not isinstance(project, Mapping):
        raise ValueError("manifest is missing project source provenance")
    project_required = ("path", "count", "source_url", "provenance")
    missing_project = [key for key in project_required if not project.get(key)]
    if missing_project:
        raise ValueError(f"manifest is missing project source provenance: {', '.join(missing_project)}")
    if int(project["count"]) != MODE_CASE_COUNTS["smoke"]:
        raise ValueError(f"smoke manifest must contain exactly {MODE_CASE_COUNTS['smoke']} cases")

    hf = manifest.get("huggingface")
    if not isinstance(hf, Mapping):
        raise ValueError("manifest is missing Hugging Face provenance")
    required = (
        "id", "config", "revision", "license", "language", "source_url",
        "metadata_api", "rows_api", "splits",
    )
    missing = [key for key in required if not hf.get(key)]
    if missing:
        raise ValueError(f"manifest is missing Hugging Face provenance: {', '.join(missing)}")
    if not isinstance(hf["splits"], Mapping):
        raise ValueError("manifest Hugging Face splits must be an object")
    for split_name in ("improvement", "validation"):
        split_info = hf["splits"].get(split_name)
        if not isinstance(split_info, Mapping):
            raise ValueError(f"manifest is missing {split_name} split")
        required_split = ("source_split", "count", "selection")
        missing_split = [key for key in required_split if not split_info.get(key)]
        if missing_split:
            raise ValueError(f"manifest {split_name} split is missing: {', '.join(missing_split)}")
        selection = split_info.get("selection", [])
        if not isinstance(selection, list) or len(selection) != int(split_info["count"]):
            raise ValueError(f"manifest count mismatch for {split_name}")
        if int(split_info["count"]) != MODE_CASE_COUNTS[split_name]:
            raise ValueError(f"{split_name} manifest must contain exactly {MODE_CASE_COUNTS[split_name]} cases")
        if any(not entry.get("row_id") or len(entry.get("row_hash", "")) != 64 for entry in selection):
            raise ValueError(f"manifest has incomplete immutable row identity for {split_name}")
        row_ids = [str(entry["row_id"]) for entry in selection]
        row_indices = [int(entry["row_idx"]) for entry in selection]
        if len(row_ids) != len(set(row_ids)) or len(row_indices) != len(set(row_indices)):
            raise ValueError(f"manifest has duplicate selected rows for {split_name}")
    validate_split_disjointness(manifest)
    return manifest
