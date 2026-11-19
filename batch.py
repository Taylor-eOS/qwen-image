import sys
from pathlib import Path

INPUT_FILE = Path(__file__).resolve().parent / "input.txt"

def load_prompts():
    if not INPUT_FILE.is_file():
        sys.exit(f"Input file not found: {INPUT_FILE}")
    lines = INPUT_FILE.read_text(encoding="utf-8-sig").splitlines() + [""]
    prompts = []
    current = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            prompts.append(" ".join(current))
            current = []
    if not prompts:
        sys.exit(f"No prompts found in {INPUT_FILE}")
    return prompts

def choose_prompts(label):
    answer = input(f"{label} (use prompt file): ").strip()
    if answer:
        return [answer]
    return load_prompts()

def run_all(prompts, job):
    failed = []
    for number, prompt in enumerate(prompts, 1):
        print(f"\n=== Prompt {number} of {len(prompts)}: {prompt[:80]} ===")
        if not job(prompt):
            failed.append(number)
    if failed:
        sys.exit(f"Prompts that failed: {', '.join(str(number) for number in failed)}")
    print(f"\nAll {len(prompts)} prompts finished")
