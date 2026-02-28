#!/usr/bin/env python3
"""Run NMF experiments using scikit-learn.

This script trains NMF models with K in {5, 10, 15, 20, 25} topics
using the same preprocessing as LDA for objective comparison.
"""

import sys
import pickle
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gensim.corpora import Dictionary

from topic_analysis.models.nmf_model import train_nmf, get_nmf_topics
from topic_analysis.evaluation.coherence import compute_coherence
from topic_analysis.evaluation.diversity import compute_topic_diversity
from topic_analysis.config import TOPIC_COUNTS, MODELS_DIR, RESULTS_DIR


def main():
    print("=" * 60)
    print("NMF (scikit-learn) Experiments")
    print("=" * 60)

    # Load preprocessed data (same as LDA)
    preprocessed_path = MODELS_DIR / "preprocessed_data.pkl"
    if not preprocessed_path.exists():
        print(f"Error: Preprocessed data not found. Run 02_preprocess.py first.")
        sys.exit(1)

    print("\nLoading preprocessed data...")
    with open(preprocessed_path, "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    corpus = data["corpus"]
    dictionary = Dictionary.load(data["dictionary_path"])

    print(f"  Documents: {len(corpus):,}")
    print(f"  Vocabulary: {len(dictionary):,}")

    # Results storage
    all_results = {}

    # Train models with different topic counts
    for num_topics in TOPIC_COUNTS:
        print(f"\n{'='*40}")
        print(f"Training NMF with K={num_topics} topics")
        print("=" * 40)

        try:
            # Train model using sklearn NMF
            model, dtm, _ = train_nmf(
                corpus=corpus,
                dictionary=dictionary,
                num_topics=num_topics,
                save_path=MODELS_DIR / f"nmf_k{num_topics}.pkl",
            )

            # Get formatted topics
            topics = get_nmf_topics(model, dictionary, num_words=10)

            # Compute coherence metrics using topic word lists
            print("\nComputing coherence metrics...")
            topic_words = [t["words"] for t in topics]
            coherence_cv = compute_coherence(topic_words, texts, dictionary, "c_v")
            coherence_npmi = compute_coherence(topic_words, texts, dictionary, "c_npmi")
            coherence_umass = compute_coherence(topic_words, texts, dictionary, "u_mass")

            # Compute diversity
            diversity = compute_topic_diversity(topic_words)

            print(f"\nResults for K={num_topics}:")
            print(f"  C_v coherence: {coherence_cv:.4f}")
            print(f"  NPMI coherence: {coherence_npmi:.4f}")
            print(f"  UMass coherence: {coherence_umass:.4f}")
            print(f"  Diversity: {diversity:.4f}")

            # Print topics
            print(f"\nTopics:")
            for topic in topics[:5]:
                print(f"  Topic {topic['id']}: {topic['formatted']}")
            if len(topics) > 5:
                print(f"  ... and {len(topics) - 5} more topics")

            # Store results
            all_results[num_topics] = {
                "num_topics": num_topics,
                "topics": topics,
                "coherence": {
                    "c_v": coherence_cv,
                    "c_npmi": coherence_npmi,
                    "u_mass": coherence_umass,
                },
                "diversity": diversity,
            }

        except Exception as e:
            print(f"Error training NMF with K={num_topics}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if all_results:
        results_path = RESULTS_DIR / "nmf_results.json"
        print(f"\n\nSaving results to: {results_path}")

        json_results = {}
        for k, v in all_results.items():
            json_results[str(k)] = v

        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
