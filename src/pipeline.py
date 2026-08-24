# ─────────────────────────────────────────────
# pipeline.py
# Generates successive "generations" of a text
# by having Gemini paraphrase/summarize the
# PREVIOUS generation's output, then measures
# Generational Drift Score (GDS) at each step.
# ─────────────────────────────────────────────

import os
import json
from datetime import datetime
from google import genai
from google.genai import types

from metrics import (
    gds,
    entropy,
    semantic_similarity,
    corpus_diversity,
)

# ─────────────────────────────────────────────
# API KEY SETUP
# ─────────────────────────────────────────────
# Set it as an environment variable in your terminal BEFORE running:
#   export GEMINI_API_KEY="your-key-here"      (Mac/Linux/bash)
#   set GEMINI_API_KEY=your-key-here            (Windows cmd)
#   $env:GEMINI_API_KEY="your-key-here"          (Windows PowerShell)
#
# This line reads it automatically — no key is ever written in this file:
client = genai.Client(api_key=os.environ.get("AQ.Ab8RN6LkJMOt6x50EgXGDvDTO4KjOhKG3oLeZ3sSjDzD2-_dGw"))

MODEL = "gemini-3.6-flash"
TEMPERATURE = 1.3   # 0.0 = deterministic/robotic, 2.0 = max randomness
                     # 1.2–1.4 gives noticeably more drift per generation


def generate_next_generation(previous_text: str) -> str:
    """
    Ask Gemini to paraphrase/summarize the previous
    generation's text — simulating one 'retelling' step,
    like a game of telephone.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=(
            "You just heard the following story once, and now you're "
            "retelling it to a friend from memory a few days later. "
            "You don't remember all the exact details anymore. "
            "Retell it in your own words, roughly half the length, "
            "focusing only on what you'd realistically still remember. "
            "It's okay to lose specific details, names, or numbers if a "
            "real person would likely forget them.\n\n"
            f"{previous_text}"
        ),
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
        ),
    )
    return response.text.strip()


def run_pipeline(original_text: str, num_generations: int = 5):
    """
    Runs the generational drift experiment.

    Returns a list of dicts, one per generation, with:
      - generation number
      - the generated text
      - cumulative GDS (vs. original)
      - step GDS (vs. immediately previous generation)
    """
    baseline_entropy = entropy(original_text)

    corpus = [original_text]          # grows every generation
    current_text = original_text
    results = []

    for gen in range(1, num_generations + 1):
        print(f"\nGenerating generation {gen}...")
        next_text = generate_next_generation(current_text)
        corpus.append(next_text)

        # Cumulative drift: original vs. this generation
        cumulative_score = gds(
            original_text, next_text, corpus, baseline_entropy
        )

        # Step drift: previous generation vs. this generation
        step_score = gds(
            current_text, next_text, [current_text, next_text], entropy(current_text)
        )

        result = {
            "generation": gen,
            "timestamp": datetime.now().isoformat(),
            "text": next_text,
            "gds_cumulative": cumulative_score,
            "gds_step": step_score,
        }
        results.append(result)

        print(f"  Text: {next_text[:80]}...")
        print(f"  Cumulative GDS: {cumulative_score}")
        print(f"  Step GDS:       {step_score}")

        current_text = next_text  # next round builds on THIS output

    return results


def save_results(results: list, path: str = "results/pipeline_log.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} generations to {path}")


if __name__ == "__main__":

    original = """NASA scientist Dr. Sarah Chen discovered water 
    on Mars on June 12 2024 using the Perseverance rover's 
    spectrometer. The finding confirms long-held theories 
    about ancient Martian oceans."""

    print("=== Running Generational Drift Pipeline ===")
    print(f"Original: {original.strip()}\n")

    results = run_pipeline(original, num_generations=5)
    save_results(results)

    print("\n=== Summary ===")
    for r in results:
        print(f"Gen {r['generation']}: cumulative={r['gds_cumulative']}  step={r['gds_step']}")