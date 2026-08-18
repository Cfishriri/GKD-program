import json
from pathlib import Path


def validate_record(record: dict) -> dict[str, str]:
    question = str(record.get("question", "")).strip()
    answer = str(record.get("answer", "")).strip()
    if not question:
        raise ValueError("record.question must be a non-empty string")
    if not answer:
        raise ValueError("record.answer must be a non-empty string")
    return {"question": question, "answer": answer}


def load_records(path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path, columns=["question", "answer"]).to_dict("records")
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    else:
        raise ValueError(f"unsupported dataset format: {path.suffix}")
    records = [validate_record(row) for row in rows]
    return records[:limit] if limit is not None else records

