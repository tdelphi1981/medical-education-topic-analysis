#!/usr/bin/env python3
"""Run BERTopic experiments.

This script trains BERTopic models with automatic and fixed topic counts.
"""

import sys
import pickle
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from topic_analysis.models.bertopic_model import train_bertopic, get_bertopic_topics
from topic_analysis.config import TOPIC_COUNTS, MODELS_DIR, RESULTS_DIR


def main():
    print("=" * 60)
    print("BERTopic Experiments")
    print("=" * 60)

    # Load raw texts
    raw_texts_path = MODELS_DIR / "raw_texts.pkl"
    if not raw_texts_path.exists():
        print(f"Error: Raw texts not found. Run 02_preprocess.py first.")
        sys.exit(1)

    print("\nLoading documents...")
    with open(raw_texts_path, "rb") as f:
        documents = pickle.load(f)

    # Filter out empty documents
    documents = [doc for doc in documents if doc and doc.strip()]
    print(f"  Documents: {len(documents):,}")

    # Results storage
    all_results = {}

    # First, train with automatic topic detection
    print(f"\n{'='*40}")
    print("Training BERTopic (automatic topic count)")
    print("=" * 40)

    try:
        model, topics, probs = train_bertopic(
            documents=documents,
            nr_topics=None,  # Automatic
            save_path=MODELS_DIR / "bertopic_auto.model",
        )

        # Get formatted topics
        formatted_topics = get_bertopic_topics(model, num_words=10)

        num_topics = len(formatted_topics)
        print(f"\nTopics discovered: {num_topics}")

        # Print sample topics
        print(f"\nSample Topics:")
        for topic in formatted_topics[:5]:
            print(f"  Topic {topic['id']} ({topic.get('count', '?')} docs): {topic['formatted']}")
        if len(formatted_topics) > 5:
            print(f"  ... and {len(formatted_topics) - 5} more topics")

        all_results["auto"] = {
            "num_topics": num_topics,
            "topics": formatted_topics,
        }

        # Now try reducing to specific topic counts
        # Process in descending order and reload model each time to get fresh copy
        for target_topics in sorted(TOPIC_COUNTS, reverse=True):
            if target_topics < num_topics:
                print(f"\n{'='*40}")
                print(f"Reducing to K={target_topics} topics")
                print("=" * 40)

                # Reload the original model to start fresh each time
                from bertopic import BERTopic
                fresh_model = BERTopic.load(str(MODELS_DIR / "bertopic_auto.model"))

                # Request +1 to account for outlier topic (-1) not being counted
                fresh_model.reduce_topics(documents, nr_topics=target_topics + 1)

                reduced_topics = get_bertopic_topics(fresh_model, num_words=10)
                print(f"Topics after reduction: {len(reduced_topics)}")

                # Save reduced model
                fresh_model.save(str(MODELS_DIR / f"bertopic_k{target_topics}.model"))

                all_results[target_topics] = {
                    "num_topics": len(reduced_topics),
                    "topics": reduced_topics,
                }
            else:
                print(f"\nSkipping K={target_topics} (more than discovered topics)")

    except ImportError as e:
        print(f"Error: BERTopic dependencies not available: {e}")
        print("Install with: pip install bertopic sentence-transformers umap-learn hdbscan")
        sys.exit(1)
    except Exception as e:
        print(f"Error training BERTopic: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save results
    results_path = RESULTS_DIR / "bertopic_results.json"
    print(f"\n\nSaving results to: {results_path}")

    json_results = {}
    for k, v in all_results.items():
        json_results[str(k)] = v

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
