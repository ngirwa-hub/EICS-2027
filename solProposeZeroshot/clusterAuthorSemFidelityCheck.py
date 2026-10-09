#!/usr/bin/env python3
"""
Measure semantic fidelity between authored final cluster writeups and the
source cluster summaries CSV after applying the author's cluster merges.

Scoring design:
- key_phrases candidate vs merged sample_titles + summary reference
- objective + description candidate vs merged explanations_text reference

Outputs:
- semantic-fidelity/cluster_author_semantic_fidelity.csv
- semantic-fidelity/cluster_author_semantic_fidelity.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from bert_score import score as bertscore_score
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: bert-score. Install it first, for example with "
        "`/Users/HP/Documents/.venv312/bin/pip install bert-score`."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = SCRIPT_DIR / "clustering_zeroshot" / "cluster_summaries_zeroshot.csv"
DEFAULT_AUTHOR_FILE = SCRIPT_DIR / "author-final.md"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "semantic-fidelity"

SCORE_TARGETS = ["key_phrases", "objective_description"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BERTScore semantic fidelity for authored cluster summaries."
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=DEFAULT_SOURCE_CSV,
        help=f"Path to cluster_summaries_zeroshot.csv. Default: {DEFAULT_SOURCE_CSV}",
    )
    parser.add_argument(
        "--author-file",
        type=Path,
        default=DEFAULT_AUTHOR_FILE,
        help=f"Path to author-final.md. Default: {DEFAULT_AUTHOR_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for semantic fidelity outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model-type",
        default="roberta-large",
        help="Backbone model for BERTScore. Default: roberta-large",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for BERTScore. Default: 8",
    )
    parser.add_argument(
        "--rescale-with-baseline",
        action="store_true",
        help="Use BERTScore baseline rescaling.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_cluster_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {row["cluster_id"].strip(): row for row in reader}
    if not rows:
        raise ValueError(f"No cluster rows found in CSV: {path}")
    return rows


def parse_author_blocks(markdown: str) -> List[Dict[str, object]]:
    blocks = [block.strip() for block in re.split(r"\n(?=# solution:)", markdown.strip()) if block.strip()]
    parsed: List[Dict[str, object]] = []

    for block in blocks:
        header_match = re.search(r"^# solution:\s*(.+?)\s*\(([^)]+)\)\s*$", block, flags=re.MULTILINE)
        if not header_match:
            raise ValueError(f"Could not parse solution header in block:\n{block[:300]}")

        solution_type = header_match.group(1).strip()
        cluster_spec = header_match.group(2).strip()
        cluster_ids = [part.strip() for part in re.findall(r"\d+", cluster_spec)]
        if not cluster_ids:
            raise ValueError(f"No cluster ids found in header: {cluster_spec}")

        fields = {
            "solution_name": "",
            "key_phrases": "",
            "objective": "",
            "description": "",
        }
        for key in fields:
            pattern = rf"^##\s*{re.escape(key).replace('_', '[_ ]')}\s*:\s*(.+?)\s*$"
            match = re.search(pattern, block, flags=re.MULTILINE | re.IGNORECASE)
            if not match:
                raise ValueError(f"Missing field `{key}` in block:\n{block[:500]}")
            fields[key] = match.group(1).strip()

        parsed.append(
            {
                "solution_type": solution_type,
                "cluster_spec": cluster_spec,
                "cluster_ids": cluster_ids,
                **fields,
            }
        )
    return parsed


def merge_reference_rows(author_block: Dict[str, object], cluster_rows: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    cluster_ids = author_block["cluster_ids"]
    merged_rows = []
    for cluster_id in cluster_ids:
        if cluster_id not in cluster_rows:
            raise ValueError(f"Cluster id {cluster_id} referenced in author-final.md not found in source CSV.")
        merged_rows.append(cluster_rows[cluster_id])

    sample_titles = " | ".join(row["sample_titles"].strip() for row in merged_rows if row["sample_titles"].strip())
    summaries = " ".join(row["summary"].strip() for row in merged_rows if row["summary"].strip())
    explanations = "\n".join(row["explanations_text"].strip() for row in merged_rows if row["explanations_text"].strip())

    return {
        "key_phrases_reference": normalize_text(f"{sample_titles}\n\n{summaries}"),
        "objective_description_reference": normalize_text(explanations),
    }


def score_pairs(
    candidates: List[str],
    references: List[str],
    model_type: str,
    batch_size: int,
    rescale_with_baseline: bool,
):
    precision, recall, f1 = bertscore_score(
        candidates,
        references,
        lang="en",
        model_type=model_type,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )
    return precision, recall, f1


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    args = parse_args()

    cluster_rows = load_cluster_rows(args.source_csv)
    author_blocks = parse_author_blocks(read_text(args.author_file))

    rows: List[Dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for block_index, author_block in enumerate(author_blocks, start=1):
        refs = merge_reference_rows(author_block, cluster_rows)

        candidates = [
            normalize_text(author_block["key_phrases"]),
            normalize_text(f"{author_block['objective']}\n\n{author_block['description']}"),
        ]
        references = [
            refs["key_phrases_reference"],
            refs["objective_description_reference"],
        ]

        print(
            f"Scoring solution {block_index}: clusters {', '.join(author_block['cluster_ids'])}"
        )

        precision, recall, f1 = score_pairs(
            candidates=candidates,
            references=references,
            model_type=args.model_type,
            batch_size=args.batch_size,
            rescale_with_baseline=args.rescale_with_baseline,
        )

        for idx, score_target in enumerate(SCORE_TARGETS):
            rows.append(
                {
                    "solution_index": block_index,
                    "solution_type": author_block["solution_type"],
                    "source_clusters": ",".join(author_block["cluster_ids"]),
                    "solution_name": author_block["solution_name"],
                    "score_target": score_target,
                    "precision": round(float(precision[idx]), 6),
                    "recall": round(float(recall[idx]), 6),
                    "f1": round(float(f1[idx]), 6),
                }
            )

    aggregate_rows = []
    for score_target in SCORE_TARGETS:
        subset = [row for row in rows if row["score_target"] == score_target]
        aggregate_rows.append(
            {
                "solution_index": "ALL",
                "solution_type": "aggregate",
                "source_clusters": "ALL",
                "solution_name": "ALL",
                "score_target": score_target,
                "precision": round(mean(float(row["precision"]) for row in subset), 6),
                "recall": round(mean(float(row["recall"]) for row in subset), 6),
                "f1": round(mean(float(row["f1"]) for row in subset), 6),
            }
        )

    csv_path = args.output_dir / "cluster_author_semantic_fidelity.csv"
    json_path = args.output_dir / "cluster_author_semantic_fidelity.json"

    all_rows = rows + aggregate_rows
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "solution_index",
                "solution_type",
                "source_clusters",
                "solution_name",
                "score_target",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    payload = {
        "source_csv": str(args.source_csv.resolve()),
        "author_file": str(args.author_file.resolve()),
        "model_type": args.model_type,
        "rescale_with_baseline": args.rescale_with_baseline,
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
