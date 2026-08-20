import argparse
import json
import os
import tempfile
from pathlib import Path

from framework_opd.framework_validation import audit_framework_records


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def audit_jsonl(path: str | Path) -> dict:
    records: list[dict] = []
    invalid_json_lines: list[int] = []
    non_object_lines: list[int] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines.append(line_number)
                continue
            if not isinstance(value, dict):
                non_object_lines.append(line_number)
                continue
            records.append(value)

    audit = audit_framework_records(records)
    malformed = len(invalid_json_lines) + len(non_object_lines)
    if malformed:
        audit["total"] += malformed
        audit["invalid"] += malformed
        reasons = dict(audit["reasons"])
        if invalid_json_lines:
            reasons["invalid_json"] = len(invalid_json_lines)
        if non_object_lines:
            reasons["record_not_object"] = len(non_object_lines)
        audit["reasons"] = dict(sorted(reasons.items()))
        examples = list(audit["invalid_examples"])
        examples.extend(
            {"line": line_number, "reasons": ["invalid_json"]}
            for line_number in invalid_json_lines[: max(0, 20 - len(examples))]
        )
        examples.extend(
            {"line": line_number, "reasons": ["record_not_object"]}
            for line_number in non_object_lines[: max(0, 20 - len(examples))]
        )
        audit["invalid_examples"] = examples[:20]
        audit["leakage_rate"] = audit["leakage_records"] / audit["total"] if audit["total"] else 0.0
    audit["data"] = str(path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit framework-label structure and answer leakage")
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--output",
        "--audit-output",
        dest="output",
        help="Audit JSON path (default: <data stem>.audit.json)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output = Path(args.output) if args.output else data_path.with_name(data_path.stem + ".audit.json")
    audit = audit_jsonl(data_path)
    _atomic_write_json(output, audit)
    print(json.dumps(audit, ensure_ascii=False))
    if audit["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
