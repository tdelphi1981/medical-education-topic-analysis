#!/usr/bin/env python3
"""Evaluate and compare all trained models.

This script computes coherence metrics for NMF and BERTopic results
using the same methodology as LDA, enabling fair comparison.
"""

import sys
import pickle
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gensim.corpora import Dictionary

from topic_analysis.evaluation.coherence import compute_all_coherence_metrics
from topic_analysis.evaluation.diversity import compute_topic_diversity
from topic_analysis.config import MODELS_DIR, RESULTS_DIR


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    if not results_path.exists():
        return {}
    with open(results_path, "r") as f:
        return json.load(f)


def filter_oov_words(topic_words: list[list[str]], dictionary: Dictionary) -> list[list[str]]:
    """Filter out-of-vocabulary words from topics.

    Args:
        topic_words: List of topics, each a list of words.
        dictionary: Gensim dictionary with valid vocabulary.

    Returns:
        Filtered topics with only in-vocabulary words.
    """
    vocab = set(dictionary.token2id.keys())
    filtered = []
    for topic in topic_words:
        filtered_topic = [w for w in topic if w in vocab]
        filtered.append(filtered_topic)
    return filtered


def main():
    print("=" * 60)
    print("Model Evaluation and Comparison")
    print("=" * 60)

    # Load preprocessed data for coherence computation
    preprocessed_path = MODELS_DIR / "preprocessed_data.pkl"
    if not preprocessed_path.exists():
        print(f"Error: Preprocessed data not found. Run 02_preprocess.py first.")
        sys.exit(1)

    print("\nLoading preprocessed data...")
    with open(preprocessed_path, "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    dictionary = Dictionary.load(data["dictionary_path"])

    # Load results from each model
    lda_results = load_results(RESULTS_DIR / "lda_results.json")
    nmf_results = load_results(RESULTS_DIR / "nmf_results.json")
    bertopic_results = load_results(RESULTS_DIR / "bertopic_results.json")

    # Evaluate NMF topics (if not already evaluated)
    print("\n" + "=" * 40)
    print("Evaluating NMF Topics")
    print("=" * 40)

    for k, result in nmf_results.items():
        if "coherence" not in result or result.get("coherence") is None:
            topics = result.get("topics", [])
            if topics:
                topic_words = [t.get("words", []) for t in topics]
                if topic_words and topic_words[0]:
                    print(f"\nComputing coherence for NMF K={k}...")
                    coherence = compute_all_coherence_metrics(topic_words, texts, dictionary)
                    diversity = compute_topic_diversity(topic_words)

                    nmf_results[k]["coherence"] = coherence
                    nmf_results[k]["diversity"] = diversity

                    print(f"  C_v: {coherence.get('c_v', 'N/A')}")
                    print(f"  Diversity: {diversity:.4f}")

    # Evaluate BERTopic topics
    print("\n" + "=" * 40)
    print("Evaluating BERTopic Topics")
    print("=" * 40)

    for k, result in bertopic_results.items():
        # Always re-evaluate BERTopic (needs OOV filtering)
        if True:
            topics = result.get("topics", [])
            if topics:
                topic_words = [t.get("words", []) for t in topics]
                if topic_words and topic_words[0]:
                    print(f"\nComputing coherence for BERTopic K={k}...")
                    # Filter out-of-vocabulary words for coherence calculation
                    filtered_words = filter_oov_words(topic_words, dictionary)
                    # Only keep topics with at least 1 word after filtering
                    non_empty = [t for t in filtered_words if len(t) > 0]
                    empty_count = len(filtered_words) - len(non_empty)
                    if empty_count > 0:
                        print(f"  Note: {empty_count} topics removed (all words OOV)")
                    if non_empty:
                        coherence = compute_all_coherence_metrics(non_empty, texts, dictionary)
                    else:
                        print(f"  Warning: No valid topics for coherence")
                        coherence = {"c_v": None, "c_npmi": None, "u_mass": None}
                    diversity = compute_topic_diversity(topic_words)  # Use original for diversity

                    bertopic_results[k]["coherence"] = coherence
                    bertopic_results[k]["diversity"] = diversity

                    print(f"  C_v: {coherence.get('c_v', 'N/A')}")
                    print(f"  Diversity: {diversity:.4f}")

    # Save updated results
    if nmf_results:
        with open(RESULTS_DIR / "nmf_results.json", "w") as f:
            json.dump(nmf_results, f, indent=2)

    if bertopic_results:
        with open(RESULTS_DIR / "bertopic_results.json", "w") as f:
            json.dump(bertopic_results, f, indent=2)

    # Generate comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    print("\nC_v Coherence:")
    print("-" * 50)
    print(f"{'Model':<15} {'K=5':<10} {'K=10':<10} {'K=15':<10} {'K=20':<10} {'K=25':<10}")
    print("-" * 50)

    for model_name, results in [("LDA", lda_results), ("NMF", nmf_results), ("BERTopic", bertopic_results)]:
        row = [model_name]
        for k in ["5", "10", "15", "20", "25"]:
            if k in results and "coherence" in results[k]:
                cv = results[k]["coherence"].get("c_v")
                row.append(f"{cv:.4f}" if cv else "N/A")
            else:
                row.append("--")
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")

    print("\nDiversity:")
    print("-" * 50)
    print(f"{'Model':<15} {'K=5':<10} {'K=10':<10} {'K=15':<10} {'K=20':<10} {'K=25':<10}")
    print("-" * 50)

    for model_name, results in [("LDA", lda_results), ("NMF", nmf_results), ("BERTopic", bertopic_results)]:
        row = [model_name]
        for k in ["5", "10", "15", "20", "25"]:
            if k in results and "diversity" in results[k]:
                div = results[k]["diversity"]
                row.append(f"{div:.4f}" if div else "N/A")
            else:
                row.append("--")
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")

    # Save combined comparison
    comparison = {
        "LDA": lda_results,
        "NMF": nmf_results,
        "BERTopic": bertopic_results,
    }

    comparison_path = RESULTS_DIR / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {comparison_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
