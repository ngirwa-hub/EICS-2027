#!/usr/bin/env python3
"""Run two BERTScore checks: OpenAI -> author and source clusters -> OpenAI.

Run using /Users/HP/Documents/.venv312/bin/python compare_bertscore.py
The author's cluster memberships define alignment: combine OpenAI proposals for
those clusters, then compare key phrases separately from objective + description.
OpenAI text is the reference and human text the candidate for the first check.
The second check compares each OpenAI proposal against its source CSV cluster:
key phrases against sample_titles + summary, and objective + description against
explanations_text.

Defaults: roberta-large, its BERTScore-recommended layer, no IDF weighting, and
no baseline rescaling.
--rescale-with-baseline enables rescaling; --offline uses local model files only.
--dry-run checks the alignment without loading the scoring model. Each input is
used as written; combined content is concatenated, not rewritten or deduplicated.

Output: one CSV containing both checks and per-check/per-field means. The first
check has ten detail rows; the second has sixteen. --comparison can select one.
Long inputs fail by default; --allow-truncation explicitly enables BERTScore's
normal token-limit truncation and marks affected rows. Affected scores cover
only a text prefix.
No API request, key file, or generation script is used.
Existing output files are never overwritten; use --output for another comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from statistics import fmean


HERE = Path(__file__).resolve().parent
WORKFLOW_ROOT = HERE.parent
SOURCE_ROOT = WORKFLOW_ROOT.parent
REQUIRED_FIELDS = {"solution_name", "key_phrases", "objective", "description"}
HEADER = re.compile(
    r"^# solution:\s*(single_solution|combined_solution)\s*\(cluster\s+([0-9, ]+)\)[ \t]*$",
    re.MULTILINE,
)


def parse_proposals(path: Path) -> list[dict]:
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    headers = list(HEADER.finditer(text))
    if not headers or len(headers) != len(re.findall(r"^# solution:", text, re.MULTILINE)):
        raise ValueError(f"Missing or malformed proposal headers in {path}.")
    proposals = []
    for index, header in enumerate(headers):
        clusters = tuple(sorted(int(value.strip()) for value in header[2].split(",")))
        if len(clusters) != len(set(clusters)):
            raise ValueError(f"Duplicate cluster within a proposal in {path}.")
        kind = "single_solution" if len(clusters) == 1 else "combined_solution"
        if header[1] != kind:
            raise ValueError(f"Solution type disagrees with cluster membership in {path}.")
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        fields = {}
        for match in re.finditer(r"^##\s*([a-zA-Z_ ]+):[ \t]*(.*)$", text[header.end():end], re.MULTILINE):
            key = re.sub(r"\s+", "_", match[1].strip().lower())
            if key in fields:
                raise ValueError(f"Duplicate field {key} in {path}.")
            fields[key] = match[2].strip()
        if not REQUIRED_FIELDS <= fields.keys() or any(not fields[key] for key in REQUIRED_FIELDS):
            raise ValueError(f"Missing or empty proposal fields in {path}.")
        proposals.append({"clusters": clusters, **fields})
    return proposals


def build_pairs(human: list[dict], generated: list[dict]) -> list[dict]:
    if len(human) != 5 or len(generated) != 8:
        raise ValueError("Expected five authored proposals and eight OpenAI proposals.")
    human_clusters = [cid for proposal in human for cid in proposal["clusters"]]
    by_cluster = {}
    for proposal in generated:
        if len(proposal["clusters"]) != 1:
            raise ValueError("Each OpenAI input must represent exactly one source cluster.")
        cid = proposal["clusters"][0]
        if cid in by_cluster:
            raise ValueError("OpenAI proposals contain duplicate source clusters.")
        by_cluster[cid] = proposal
    if len(set(human_clusters)) != len(human_clusters) or set(human_clusters) != set(by_cluster):
        raise ValueError("Both versions must cover the same source clusters exactly once.")
    pairs = []
    for index, proposal in enumerate(human, 1):
        matching = [by_cluster[cid] for cid in proposal["clusters"]]
        candidate_texts = {
            "key_phrases": proposal["key_phrases"],
            "objective_description": proposal["objective"] + "\n\n" + proposal["description"],
        }
        reference_texts = {
            "key_phrases": " | ".join(p["key_phrases"] for p in matching),
            "objective_description": "\n\n".join(p["objective"] + "\n\n" + p["description"] for p in matching),
        }
        for target in reference_texts:
            pairs.append({
                "comparison": "openai_reference_author_candidate",
                "proposal_index": index,
                "source_clusters": ",".join(map(str, proposal["clusters"])),
                "reference_name": " | ".join(p["solution_name"] for p in matching),
                "candidate_name": proposal["solution_name"],
                "score_target": target,
                "reference": reference_texts[target],
                "candidate": candidate_texts[target],
            })
    return pairs


def build_source_pairs(path: Path, generated: list[dict]) -> list[dict]:
    required = {"cluster_id", "cluster_name", "sample_titles", "summary", "explanations_text"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Source CSV is missing required reference fields.")
        source = {}
        for row in reader:
            cid = int(row["cluster_id"])
            if cid == -1:
                continue
            if cid < 0 or cid in source or any(not row[key] or not row[key].strip() for key in required):
                raise ValueError("Invalid, duplicated, or incomplete source cluster.")
            source[cid] = row
    if len(generated) != 8 or len(source) != 8:
        raise ValueError("Expected eight non-noise source clusters and eight OpenAI proposals.")
    if any(len(p["clusters"]) != 1 for p in generated):
        raise ValueError("Source comparison requires one OpenAI proposal per cluster.")
    ids = [p["clusters"][0] for p in generated]
    if len(set(ids)) != 8 or set(ids) != set(source):
        raise ValueError("OpenAI proposals must cover each source cluster exactly once.")
    pairs = []
    for index, proposal in enumerate(sorted(generated, key=lambda p: p["clusters"]), 1):
        cid = proposal["clusters"][0]
        row = source[cid]
        references = {
            "key_phrases": row["sample_titles"].strip() + "\n\n" + row["summary"].strip(),
            "objective_description": row["explanations_text"].strip(),
        }
        candidates = {
            "key_phrases": proposal["key_phrases"],
            "objective_description": proposal["objective"] + "\n\n" + proposal["description"],
        }
        for target in references:
            pairs.append({
                "comparison": "source_reference_openai_candidate", "proposal_index": index,
                "source_clusters": str(cid), "reference_name": row["cluster_name"],
                "candidate_name": proposal["solution_name"], "score_target": target,
                "reference": references[target], "candidate": candidates[target],
            })
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--human-file", type=Path, default=SOURCE_ROOT / "author-final.md")
    parser.add_argument("--openai-file", type=Path, default=WORKFLOW_ROOT / "generation/eight-proposals.md")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_ROOT / "clustering_zeroshot/cluster_summaries_zeroshot.csv")
    parser.add_argument("--comparison", choices=["both", "author", "source"], default="both")
    parser.add_argument("--output", type=Path, default=HERE / "bertscore-comparisons.csv")
    parser.add_argument("--model-type", default="roberta-large")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rescale-with-baseline", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-truncation", action="store_true", help="Enable standard BERTScore truncation for long texts; flag affected rows.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    generated = parse_proposals(args.openai_file)
    pairs = []
    if args.comparison in {"both", "author"}:
        pairs.extend(build_pairs(parse_proposals(args.human_file), generated))
    if args.comparison in {"both", "source"}:
        pairs.extend(build_source_pairs(args.source_csv, generated))
    for pair in pairs[::2]:
        print(f"{pair['comparison']}: clusters {pair['source_clusters']} -> {pair['candidate_name']}")
    if args.dry_run:
        print(f"Alignment checked: {len(pairs)} comparisons. No scoring model loaded or output written.")
        return 0
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("Output must be a new file in an existing directory; choose another --output.")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from bert_score import score
    from bert_score.utils import get_tokenizer
    from transformers import GPT2Tokenizer, RobertaTokenizer

    candidates = [pair["candidate"] for pair in pairs]
    references = [pair["reference"] for pair in pairs]
    tokenizer = get_tokenizer(args.model_type, use_fast=False)
    limit = tokenizer.model_max_length
    encoding_options = {"add_prefix_space": True} if isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer)) else {}
    for pair in pairs:
        pair["reference_tokens"] = len(tokenizer.encode(pair["reference"], add_special_tokens=True, truncation=False, **encoding_options))
        pair["candidate_tokens"] = len(tokenizer.encode(pair["candidate"], add_special_tokens=True, truncation=False, **encoding_options))
        pair["truncated"] = max(pair["reference_tokens"], pair["candidate_tokens"]) > limit
        pair["reference_tokens_used"] = min(pair["reference_tokens"], limit)
        pair["candidate_tokens_used"] = min(pair["candidate_tokens"], limit)
        if pair["truncated"]:
            print(f"Token limit exceeded: {pair['comparison']}, clusters {pair['source_clusters']}, {pair['score_target']} "
                  f"(reference {pair['reference_tokens']}, candidate {pair['candidate_tokens']}; limit {limit}).")
    if any(pair["truncated"] for pair in pairs) and not args.allow_truncation:
        raise ValueError("Long text detected. --allow-truncation scores only the retained text prefixes and flags affected rows.")
    print(f"Scoring {len(pairs)} pairs with {args.model_type} on {args.device}.", flush=True)
    precision, recall, f1 = score(
        candidates, references, model_type=args.model_type, lang="en",
        batch_size=args.batch_size, device=args.device, idf=False,
        rescale_with_baseline=args.rescale_with_baseline, verbose=False,
    )
    settings = {"model_type": args.model_type, "rescale_with_baseline": args.rescale_with_baseline}
    rows = []
    for index, pair in enumerate(pairs):
        row = {key: value for key, value in pair.items() if key not in {"reference", "candidate"}}
        row.update(precision=float(precision[index]), recall=float(recall[index]), f1=float(f1[index]), **settings)
        rows.append(row)
    for comparison in dict.fromkeys(pair["comparison"] for pair in pairs):
        for target in ("key_phrases", "objective_description"):
            subset = [row for row in rows if row["comparison"] == comparison and row["score_target"] == target]
            summary = {
                "comparison": comparison, "proposal_index": "MEAN", "source_clusters": "ALL",
                "reference_name": "ALL", "candidate_name": "ALL", "score_target": target,
                "reference_tokens": "", "candidate_tokens": "", "truncated": any(row["truncated"] for row in subset),
                "reference_tokens_used": "", "candidate_tokens_used": "",
                **{metric: fmean(row[metric] for row in subset) for metric in ("precision", "recall", "f1")},
                **settings,
            }
            rows.append(summary)
            print(f"Mean {comparison}, {target}: P={summary['precision']:.6f}, R={summary['recall']:.6f}, F1={summary['f1']:.6f}")
    with args.output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()})
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
