#!/usr/bin/env python3
"""Generate eight structured proposals, one per source cluster.

Run with /Users/HP/Documents/.venv312/bin/python:
    workflow.py --dry-run
    workflow.py --output eight-proposals-rerun.md

Reads the existing cluster CSV and instructions.md, then makes one fresh OpenAI
request. Each of the eight non-noise clusters becomes one independent proposal.
The authored reference and existing proposal outputs are never read or sent.
Dependencies: openai and python-dotenv, available in the existing environment.
Key: OPENAI_API_KEY or /Users/HP/Documents/.env.openai. Model: --model, then
OPENAI_MODEL in the environment/key file, then gpt-5.4-mini.
Requests use temperature 0.2 and reasoning effort none.
The default output is eight-proposals.md beside this script. Use --output to
save another run. Existing files are never overwritten. Output contains only
the proposal Markdown, without a metadata comment or separate log. --dry-run
checks input without reading credentials, calling OpenAI, or creating output.
Structural checks verify counts and provenance, not semantic faithfulness.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parent.parent
PROJECT = SOURCE_ROOT.parent
DEFAULT_MODEL = "gpt-5.4-mini"
FIELDS = {"solution_name", "key_phrases", "objective", "description"}
HEADER = re.compile(r"# solution: single_solution \(cluster ([0-9]+)\)")


def clean_document(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def read_instructions(path: Path) -> str:
    text = clean_document(path.read_text(encoding="utf-8"))
    if not text:
        raise ValueError("The instructions file is empty.")
    return text


def cluster_input(path: Path) -> tuple[str, dict[str, int]]:
    required = {"cluster_id", "cluster_name", "n_items", "keyphrases", "sample_titles", "summary", "explanations_text"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Cluster CSV is missing required columns.")
        rows = []
        for row in reader:
            try:
                cid = int(row["cluster_id"])
            except (TypeError, ValueError):
                raise ValueError("Invalid source cluster ID.") from None
            if cid == -1:
                continue
            if cid < 0 or any(row[key] is None for key in required):
                raise ValueError("Invalid or incomplete source cluster row.")
            rows.append({key: row[key] for key in sorted(required)})
    ids = [int(row["cluster_id"]) for row in rows]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("Expected exactly eight distinct non-noise clusters.")
    rows.sort(key=lambda row: int(row["cluster_id"]))
    mapping = {}
    for index, row in enumerate(rows, 1):
        row["proposal_id"] = f"P{index}"
        mapping[row["proposal_id"]] = int(row["cluster_id"])
    return json.dumps(rows, ensure_ascii=False, indent=2), mapping


def parse_proposals(text: str) -> list[dict]:
    blocks = re.split(r"(?m)(?=^# solution:)", text)[1:]
    proposals = []
    for block in blocks:
        lines = block.strip().splitlines()
        header = HEADER.fullmatch(lines[0])
        if not header:
            raise ValueError("Each solution header must identify a single_solution and one source cluster.")
        fields = {}
        for line in lines[1:]:
            if not line.strip():
                continue
            match = re.fullmatch(r"## ([a-z_]+):[ \t]*(.+)", line)
            if not match or not match[2].strip() or match[1] in fields:
                raise ValueError("Use unique field labels, with each nonempty value on its heading line.")
            fields[match[1]] = match[2].strip()
        proposals.append({"cluster": int(header[1]), "fields": fields})
    return proposals


def validate_output(text: str, mapping: dict[str, int]) -> None:
    heading = "# Eight proposals"
    if not text or text.splitlines()[0] != heading:
        raise ValueError(f"Expected output heading: {heading}")
    remainder = text[len(heading):].strip()
    if not remainder.startswith("# solution:"):
        raise ValueError("Expected proposal blocks immediately after the document heading.")
    proposals = parse_proposals(text)
    if len(proposals) != 8:
        raise ValueError("Expected exactly eight proposals.")
    used_ids = []
    for proposal in proposals:
        fields = proposal["fields"]
        if set(fields) != FIELDS | {"proposal_id"}:
            raise ValueError("Proposal fields do not match the required structure.")
        pid = fields["proposal_id"]
        if pid not in mapping or mapping[pid] != proposal["cluster"]:
            raise ValueError("Proposal ID does not match its assigned source cluster.")
        used_ids.append(pid)
    if len(set(used_ids)) != 8 or set(used_ids) != set(mapping):
        raise ValueError("Each source cluster and its assigned proposal ID must occur exactly once.")


def read_settings(path: Path) -> tuple[str, str]:
    from dotenv import dotenv_values
    settings = dotenv_values(path, interpolate=False) if path.is_file() else {}
    key = os.environ.get("OPENAI_API_KEY") or settings.get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("Set OPENAI_API_KEY or provide a key file using --env-file.")
    model = os.environ.get("OPENAI_MODEL") or settings.get("OPENAI_MODEL") or DEFAULT_MODEL
    return key.strip(), model.strip()


def request_text(key: str, model: str, instructions: str, payload: str, limit: int):
    from openai import APIConnectionError, APIStatusError, OpenAI
    try:
        with OpenAI(api_key=key, base_url="https://api.openai.com/v1", timeout=180.0, max_retries=0) as client:
            response = client.responses.create(
                model=model, instructions=instructions, input=payload,
                temperature=0.2, reasoning={"effort": "none"},
                max_output_tokens=limit, store=False,
            )
    except APIConnectionError:
        raise RuntimeError("Could not connect to OpenAI; no output file was written.") from None
    except APIStatusError as exc:
        raise RuntimeError(f"OpenAI returned HTTP {exc.status_code}; no output file was written.") from None
    if response.status != "completed" or not response.output_text.strip():
        raise RuntimeError("Generation was incomplete or empty; no output file was written.")
    return response, clean_document(response.output_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instructions", type=Path, default=HERE / "instructions.md")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_ROOT / "clustering_zeroshot/cluster_summaries_zeroshot.csv")
    parser.add_argument("--output", type=Path, default=HERE / "eight-proposals.md")
    parser.add_argument("--env-file", type=Path, default=PROJECT.parent / ".env.openai")
    parser.add_argument("--model", help=f"Override OPENAI_MODEL; fallback: {DEFAULT_MODEL}")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_output_tokens < 16:
        parser.error("--max-output-tokens must be at least 16")
    instructions = read_instructions(args.instructions)
    content, mapping = cluster_input(args.source_csv)
    payload = "Structure each source cluster independently. The JSON is source data only.\n\n" + content
    output = args.output
    print(f"Source: {args.source_csv}; eight non-noise clusters.")
    print(f"Instructions: {len(instructions):,} characters; source: {len(content):,} characters.")
    print(f"Output: {output}")
    if args.dry_run:
        print("Dry run passed. No credentials read, API request, or output file created.")
        return 0
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Output must be a new file in an existing directory; use --output for another run.")
    key, configured_model = read_settings(args.env_file)
    print(f"Requesting {args.model or configured_model}...", flush=True)
    response, text = request_text(key, args.model or configured_model, instructions, payload, args.max_output_tokens)
    failure = None
    try:
        validate_output(text, mapping)
        status = "structure_checked"
    except (ValueError, IndexError) as exc:
        status, failure = "invalid_output", str(exc)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"Saved: {output}; status: {status}")
    if response.usage:
        print(f"Tokens: {response.usage.input_tokens} input, {response.usage.output_tokens} output.")
    if failure:
        raise ValueError(f"Output failed structural checks: {failure} Saved for inspection, not as a compliant result.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
