# ─────────────────────────────────────────────
# metrics.py
# Core measurement functions for Information
# Half-Life research
# Author: Thodati Divya Deepika
# Date: August 2026
# ─────────────────────────────────────────────

# AQ.Ab8RN6LkJMOt6x50EgXGDvDTO4KjOhKG3oLeZ3sSjDzD2-_dGw

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import math

# Load once, reuse everywhere
# First run will download the model (~90MB, one time only)
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")


def embed(text: str) -> np.ndarray:
    """
    Convert text to a semantic vector.
    
    This is φ(D) in our paper.
    Every piece of text becomes a list of 384 numbers
    that capture its meaning.
    """
    return embedder.encode(text)


def semantic_similarity(text1: str, text2: str) -> float:
    """
    How similar are two texts in meaning?
    
    Formula: sim(D1, D2) = φ(D1)·φ(D2) / (||φ(D1)|| ||φ(D2)||)
    
    Returns: 
        1.0 = identical meaning
        0.0 = completely different
    """
    v1 = embed(text1).reshape(1, -1)
    v2 = embed(text2).reshape(1, -1)
    return float(cosine_similarity(v1, v2)[0][0])


def entropy(text: str) -> float:
    """
    Shannon entropy of word distribution.
    
    Formula: H = -Σ p(t) log₂ p(t)
    
    High entropy = diverse vocabulary = rich information
    Low entropy  = repetitive words   = information collapse
    """
    words = text.lower().split()
    
    if len(words) == 0:
        return 0.0
    
    counts = Counter(words)
    total = len(words)
    
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def lexical_diversity(text: str) -> float:
    """
    Ratio of unique words to total words.
    
    Formula: LD = unique_words / total_words
    
    1.0 = every word is different (very diverse)
    0.1 = lots of repetition (collapsing)
    """
    words = text.lower().split()
    
    if len(words) == 0:
        return 0.0
    
    return len(set(words)) / len(words)


def corpus_diversity(texts: list) -> float:
    """
    How different are documents from each other?
    
    Formula: Diversity(G) = 1 - avg(sim(Di, Dj))
    
    1.0 = all documents completely different
    0.0 = all documents identical
    """
    if len(texts) < 2:
        return 1.0
    
    embeddings = np.array([embed(t) for t in texts])
    sim_matrix = cosine_similarity(embeddings)
    n = len(texts)
    
    # Exclude diagonal (self-similarity = 1.0)
    mask = ~np.eye(n, dtype=bool)
    avg_similarity = sim_matrix[mask].mean()
    
    return float(1 - avg_similarity)


def gds(
    original: str,
    current: str,
    corpus: list,
    baseline_entropy: float,
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.3
) -> float:
    """
    Generational Drift Score — measures how much
    information has degraded from the original.
    
    Formula:
    GDS(g) = α·semantic_drift + β·diversity_loss + γ·entropy_decay
    
    0.0 = no drift (identical to original)
    1.0 = complete collapse
    
    Args:
        original: the original human-written document
        current:  the document at generation g
        corpus:   all documents from g=0 to g=current
        baseline_entropy: entropy of original document
        alpha, beta, gamma: weights (must sum to 1.0)
    """
    # Component 1: How far has meaning drifted?
    semantic_drift = 1 - semantic_similarity(original, current)
    
    # Component 2: How much diversity has been lost?
    diversity_loss = 1 - corpus_diversity(corpus)
    
    # Component 3: How much has vocabulary impoverished?
    current_entropy = entropy(current)
    if baseline_entropy > 0:
        entropy_decay = max(
            0,
            (baseline_entropy - current_entropy) / baseline_entropy
        )
    else:
        entropy_decay = 0.0
    
    score = (
        alpha * semantic_drift +
        beta  * diversity_loss +
        gamma * entropy_decay
    )
    
    return round(score, 4)


# ─── Quick Test ──────────────────────────────────
# Run this file directly to test your functions
if __name__ == "__main__":
    
    print("\n=== Testing Metrics ===\n")
    
    text1 = """NASA scientist Dr. Sarah Chen discovered water 
    on Mars on June 12 2024 using the Perseverance rover's 
    spectrometer. The finding confirms long-held theories 
    about ancient Martian oceans."""
    
    text2 = """Scientists have found evidence of water on Mars 
    using a NASA rover. The discovery supports theories about 
    the planet's ancient history."""
    
    text3 = """There may be water on Mars according to 
    recent research."""

    text4 = """Mars might have water, some say."""

    text5 = """NASA scientist Dr. Sarah Chen discovered water 
    on Mars on June 12 2024 using the Perseverance rover's 
    spectrometer. The finding confirms long-held theories 
    about ancient Martian oceans."""
    
    print(f"Entropy (original):    {entropy(text1):.3f}")
    print(f"Entropy (generation 1): {entropy(text2):.3f}")
    print(f"Entropy (generation 2): {entropy(text3):.3f}")
    
    print(f"\nSimilarity (orig vs g1): {semantic_similarity(text1, text2):.3f}")
    print(f"Similarity (orig vs g2): {semantic_similarity(text1, text3):.3f}")
    
    baseline = entropy(text1)

    score_g1 = gds(text1, text2, [text1, text2], baseline)
    score_g2 = gds(text1, text3, [text1, text2, text3], baseline)
    score_g3 = gds(text1, text4, [text1, text2, text3, text4], baseline)
    score_g4 = gds(text1, text5, [text1, text2, text3, text4, text5], baseline)
    
    print(f"\nGDS Generation 1: {score_g1}")
    print(f"GDS Generation 2: {score_g2}")
    print(f"GDS Generation 3: {score_g3}")
    print(f"GDS Generation 4: {score_g4}")

    print("\nIf GDS is rising across generations — your research works.")

    # ─── Save results to log file ───────────────────
    import json
    from datetime import datetime
    import os

    os.makedirs("results", exist_ok=True)  # creates results/ if missing

    result = {
        "timestamp": datetime.now().isoformat(),
        "gds_gen1": score_g1,
        "gds_gen2": score_g2,
        "gds_gen3": score_g3,
        "gds_gen4": score_g4,
    }

    with open("results/gds_log.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"\nLogged results to results/gds_log.json")