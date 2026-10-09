import csv
import os
import requests
import json
import re
import datetime
from typing import List

# ==== Setup ====
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_FOLDER = "expert_responses"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
MASTER_CSV = os.path.join(OUTPUT_FOLDER, "solutionPropose_zeroshot.csv")

RAW_LOG_DIR = os.path.join(OUTPUT_FOLDER, "raw_logs_solutionPropose_zeroshot")
os.makedirs(RAW_LOG_DIR, exist_ok=True)

counter_file = "zeroshot_solutionPropose_iteration_counter.txt"

# ---- ALL VARIANTS ----
ALL_VARIANTS = [
    "llama-generalist", "llama-generalist2", "llama-normative", "llama-normative2", "llama-subject_matter", "llama-subject_matter2",
    "mistral-generalist", "mistral-generalist2", "mistral-normative", "mistral-normative2", "mistral-subject_matter", "mistral-subject_matter2",
    "phi4-generalist", "phi4-generalist2", "phi4-normative", "phi4-normative2", "phi4-subject_matter", "phi4-subject_matter2",
    "gemma3-generalist", "gemma3-generalist2", "gemma3-normative", "gemma3-normative2", "gemma3-subject_matter", "gemma3-subject_matter2",
]

# ---- Context loaders ----
def load_project():
    return (
        "- The expert elicitation is based on the Shift2DC project, which aims to accelerate the transition to a sustainable energy system by promoting the use of direct current (DC) technologies.\n"
        "- The project focuses on the development and demonstration of DC solutions in four sectors: ports, industry, data centers, and buildings."
    )

def load_solutions():
    return (
        "The proposed direct current (DC) solutions are;\n"
        " - Smart and sustainable DC cables\n"
        " - DC connectors\n"
        " - Static protection system\n"
        " - Semiconductor-based circuit breaker\n"
        " - Protection DC system design tool\n"
        " - DC-DC converter\n"
        " - LVAC-LVDC interlink converter\n"
        " - DC measurement device\n"
        " - DC solution design tool\n"
        " - Network design tool for DC solutions\n"
        " - Solid-state circuit breaker\n"
    )

def load_instructions(existing_solutions: List[str]) -> str:
    return (
        f"DO NOT repeat solutions from this list: {', '.join(existing_solutions)}\n"
        "FORMAT REQUIREMENTS:\n"
        "1. Each solution MUST be on a single line\n"
        "2. Each solution MUST have this format: \"Solution Title: Brief explanation\"\n"
        "3. The solution title must be followed by a colon, then the explanation\n"
        "\n"
        "EXAMPLE:\n"
        "DC Energy Storage Integration: A system that integrates battery storage directly with DC microgrids, eliminating AC conversion losses and improving grid stability.\n"
        "Even if you can only think of ONE valid solution, please provide it using the required format.\n"
    )

def load_question(reviewed_count: int = 11, max_solutions: int = 5) -> str:
    return (
        f"You previously reviewed a list of {reviewed_count} proposed DC solutions for the Shift2DC project.\n"
        "Please provide additional DC solution(s) that could be important for: \"ports\", \"industry\", \"data centers\", \"buildings\" "
        "and that were NOT included in the earlier list.\n\n"
        "For each solution, include:\n"
        "- Short title (max 1 line)\n"
        "- Brief explanation (1–2 lines)\n"
        "If you propose more than one solution, number them.\n"
        f"You may propose any number of additional DC solutions, up to {max_solutions}."
    )


def query_expert(model, full_prompt) -> str:
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": full_prompt},
            stream=True,
            timeout=120,
        )
    except requests.RequestException as e:
        print(f"Error connecting to model {model}: {e}")
        return ""

    if r.status_code != 200:
        print(f"Error from model {model}: {r.status_code}")
        return ""

    full_text = ""
    for line in r.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        full_text += obj.get("response", "")
    return full_text.strip()

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def parse_existing_solutions(solutions_text: str) -> List[str]:
    lines = []
    for line in solutions_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            lines.append(line[2:].strip())
    return lines

def extract_solutions(response_text: str) -> List[dict]:
    if not response_text:
        return []

    out = []
    seen = set()

    # Preferred: single line Title: Explanation
    for m in re.finditer(r"(?m)^(?:\d+[.)]\s*|[-•*]\s*)?([^:\n]{2,}?):\s*(.+)$", response_text.strip()):
        title = m.group(1).strip()
        expl = m.group(2).strip()
        key = _norm(title)
        if title and expl and key not in seen:
            out.append({"title": title, "explanation": expl})
            seen.add(key)

    if out:
        return out

    # Fallback: bullet/numbered title then explanation on next line
    for m in re.finditer(
        r"(?ms)^\s*(?:\d+[.)]\s*|[-•*]\s*)?([^\n:]{2,})\n\s*([^\n].+?)(?=\n\s*(?:\d+[.)]\s*|[-•*]\s*|$))",
        response_text.strip(),
    ):
        title = m.group(1).strip()
        expl = m.group(2).strip()
        key = _norm(title)
        if title and expl and key not in seen:
            out.append({"title": title, "explanation": expl})
            seen.add(key)
    return out

# ---- Main ----
if os.path.exists(counter_file):
    try:
        with open(counter_file, "r") as f:
            iteration_counter = int(f.read().strip())
    except ValueError:
        iteration_counter = 0
else:
    iteration_counter = 0

if __name__ == "__main__":
    # Build prompt
    solutions_text = load_solutions()
    existing_solutions = parse_existing_solutions(solutions_text)
    normalized_solutions_set = {_norm(s) for s in existing_solutions}
    context_sections = "\n\n".join([load_project(), solutions_text, load_instructions(existing_solutions)])
    question = load_question()
    full_prompt = f"{context_sections}\n\n{question}"

    structured_responses = {}
    raw_responses = {}

    # Query all variants
    for variant_name in ALL_VARIANTS:
        base_model, role = variant_name.split("-", 1)
        print(f"🔍 Querying {role} ({base_model})")
        response = query_expert(variant_name, full_prompt)
        raw_responses[variant_name] = response

        extracted = extract_solutions(response)
        # filter out overlaps with the original 11 solutions
        filtered = [it for it in extracted if _norm(it["title"]) not in normalized_solutions_set]

        if not filtered:
            print(f"⚠️ No non-overlapping structured items for {variant_name}.")
            continue

        structured_responses[variant_name] = {
            "base_model": base_model,
            "role": role,
            "items": filtered,
        }

    # Write CSV rows per variant (don’t fail the entire batch)
    rows_written = 0

    zero_shot = "zeroshot"

    # Prepare writer (create file and header if new)
    new_file = not os.path.exists(MASTER_CSV) or os.stat(MASTER_CSV).st_size == 0
    with open(MASTER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id","base_model","variant_id","model","solution_title",
                "explanation","iteration","timestamp","truncated","condition","is_duplicate"
            ],
        )
        if new_file:
            writer.writeheader()

        for vname, payload in structured_responses.items():
            base_model = payload["base_model"]
            role = payload["role"]
            items = payload["items"]

            variant_id = f"{base_model}_{role}"
            # truncated_flag = "yes" if len(items) > 5 else "no"
            truncated_flag = "no"  # we keep all returned items


            for i, b in enumerate(items, start=1):
                row_id = f"{variant_id}_{iteration_counter:02d}_b{i}"
                writer.writerow({
                    "row_id": row_id,
                    "base_model": base_model,
                    "variant_id": variant_id,
                    "model": role,
                    "solution_title": b["title"],
                    "explanation": b["explanation"],
                    "iteration": iteration_counter,
                    "timestamp": timestamp,
                    "truncated": truncated_flag,
                    "condition": zero_shot,
                    "is_duplicate": b.get("is_duplicate", "no"),
                })
                rows_written += 1

    print(f"\n✅ Appended {len(structured_responses)} variants to {MASTER_CSV}")
    print(f"📊 Wrote {rows_written} total rows across {len(structured_responses)} variants")

    # Write raw logs (once each), then bump iteration ONCE
    for vname, response_text in raw_responses.items():
        txt_path = os.path.join(RAW_LOG_DIR, f"zeroshot_solutionPropose_{vname}_{iteration_counter:02d}_{timestamp}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(response_text or "")
    print(f"📝 Wrote {len(raw_responses)} raw logs to {RAW_LOG_DIR}")

    # Advance iteration and persist
    iteration_counter += 1
    with open(counter_file, "w") as f:
        f.write(str(iteration_counter))
    print(f"🔢 Updated iteration counter: {iteration_counter}")

    # Fallback only if nothing valid was parsed
    if rows_written == 0:
        fallback_path = os.path.join(OUTPUT_FOLDER, f"zeroshot_solutionPropose_failed_{timestamp}.json")
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump({
                "question": full_prompt,
                "structured_responses": structured_responses,
                "raw_responses": raw_responses,
                "error_reason": "No non-overlapping items parsed from any model.",
            }, f, indent=4, ensure_ascii=False)
        print(f"⚠️ Fallback written to {fallback_path}")
